// Switch the heading mode of the vendor walking modules on or off in the private simulator, and keep its
// target heading up to date.
//
// WalkModuleV3 and WalkModuleV5 of the vendor controller have setHeadingMode(bool, double).  With it on, the
// yaw-rate command of the walking policy is 2 * (target heading - heading from the IMU), limited to
// +-1 rad/s; like every yaw-rate command it is dropped below 0.2 rad/s, i.e. for a heading error under
// 0.1 rad (5.7 deg).  The IMU sits in the torso, above the waist joints: the mode holds the heading of the
// torso, not of the pelvis.  A replay that turns the waist therefore has to move the target with it
// (worker.py sends the torso heading that goes with the pelvis standing at its staged heading).
//
// Nothing in this vendor build calls the function: there is no configuration key, SDK message or joystick
// binding for it.  The setter and the two vtables are exported symbols, so this library, preloaded into the
// vendor processes, finds the module objects by their vtable pointer and calls the vendor's own setter, which
// takes the module's mutex.  No vendor file is changed.
//
// The walking modules live in the controller process ("sim"), not in the MuJoCo process.  The library is
// loaded into every process of the launch; only where the setter symbol resolves does it start one thread.
// The memory search runs in that thread, so the control loop is not held up.
//
// Files in IGRIS_HEADING_DIR, all float64:
//   heading_target.bin   written by the replay: sequence, enable (0|1), target heading [rad], sum of the three
//   heading_applied.json written here when the first target was applied (or could not be)
//   heading_status.bin   written here: setter calls so far, last sequence applied, and the mode and target
//                        read back from WalkModuleV5
//
// The library does nothing unless IGRIS_HEADING_DIR is set; the launcher sets it only for the pinned
// simulator binary inside the container without network.  The read-back offsets belong to that binary.
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

namespace {
struct Module {
  const char* name; const char* vtable; const char* setter;
  long flag_offset, target_offset;      // where setHeadingMode stores its two arguments (read back only)
  std::uintptr_t object; int found;
  void (*set)(void*, bool, double);
};
Module modules[2] = {
  {"WalkModuleV3", "_ZTVN6robros10controller12WalkModuleV3E", "_ZN6robros10controller12WalkModuleV314setHeadingModeEbd", 0x5a3, 0x5a8, 0, 0, nullptr},
  {"WalkModuleV5", "_ZTVN6robros10controller12WalkModuleV5E", "_ZN6robros10controller12WalkModuleV514setHeadingModeEbd", 0x37b, 0x380, 0, 0, nullptr},
};
const std::uintptr_t MASK = 0x5a5a5a5a5a5a5a5aull;      // the plain vtable pointer is never kept in this library's memory
const double STEP = 1e-4;                               // [rad] smallest change of the target that is passed on
unsigned char page[1 << 16];

// Objects of a class: 8-byte aligned words in private writable memory equal to the class's vtable pointer.
// Returns the number of bytes read; -1 if the process maps cannot be read.
long long locate() {
  std::uintptr_t needle[2] = {0, 0};
  for (int k = 0; k < 2; k++) {
    void* table = dlsym(RTLD_DEFAULT, modules[k].vtable);
    modules[k].object = 0; modules[k].found = 0;
    modules[k].set = reinterpret_cast<void (*)(void*, bool, double)>(dlsym(RTLD_DEFAULT, modules[k].setter));
    needle[k] = table ? (reinterpret_cast<std::uintptr_t>(table) + 16) ^ MASK : 0;   // vptr: past offset-to-top and typeinfo
  }
  FILE* maps = std::fopen("/proc/self/maps", "r");
  int memory = open("/proc/self/mem", O_RDONLY);
  if (!maps || memory < 0) { if (maps) std::fclose(maps); if (memory >= 0) close(memory); return -1; }
  const std::uintptr_t own_stack = reinterpret_cast<std::uintptr_t>(&needle);
  const std::uintptr_t own_page = reinterpret_cast<std::uintptr_t>(page);
  char line[512]; long long total = 0;
  while (std::fgets(line, sizeof line, maps)) {
    unsigned long long begin, end; char permissions[8]; char path[256] = "";
    if (std::sscanf(line, "%llx-%llx %7s %*s %*s %*s %255s", &begin, &end, permissions, path) < 3) continue;
    if (std::strncmp(permissions, "rw-p", 4) || (path[0] && std::strcmp(path, "[heap]"))) continue;
    if (end - begin > (256ull << 20) || (own_stack >= begin && own_stack < end)) continue;   // not this thread's stack
    for (unsigned long long at = begin; at < end; at += sizeof page) {
      const size_t want = static_cast<size_t>(end - at < sizeof page ? end - at : sizeof page);
      if (at < own_page + sizeof page && at + want > own_page) continue;                      // not this library's buffer
      const ssize_t got = pread(memory, page, want, static_cast<off_t>(at));
      if (got <= 0) continue;
      total += got;
      for (ssize_t i = 0; i + 8 <= got; i += 8) {
        std::uintptr_t word; std::memcpy(&word, page + i, 8);
        for (int k = 0; k < 2; k++)
          if (needle[k] && (word ^ MASK) == needle[k]) { modules[k].object = at + i; modules[k].found++; }
      }
    }
  }
  std::fclose(maps); close(memory);
  return total;
}

// Call the vendor setter on every module that was identified; returns how many were set.
int apply(bool enable, double target) {
  int applied = 0;
  for (auto& module : modules)
    if (module.found == 1 && module.set) { module.set(reinterpret_cast<void*>(module.object), enable, target); applied++; }
  return applied;
}

void read_back(const Module& module, int& flag, double& stored) {
  flag = -1; stored = 0.;
  if (module.found != 1) return;
  flag = *reinterpret_cast<const unsigned char*>(module.object + module.flag_offset);
  std::memcpy(&stored, reinterpret_cast<const void*>(module.object + module.target_offset), 8);
}

void report(const std::string& directory, int enable, double target, long long scanned, const char* error) {
  const std::string temporary = directory + "/heading_applied.json.tmp";
  FILE* answer = std::fopen(temporary.c_str(), "w");
  if (!answer) std::abort();
  timespec now; clock_gettime(CLOCK_REALTIME, &now);
  std::fprintf(answer, "{\"wall\":%.6f,\"pid\":%d,\"enable\":%d,\"target_yaw_rad\":%.9f,\"scanned_bytes\":%lld,\"error\":%s%s%s,\"modules\":{",
               now.tv_sec + now.tv_nsec * 1e-9, static_cast<int>(getpid()), enable, target, scanned,
               error ? "\"" : "", error ? error : "null", error ? "\"" : "");
  for (int k = 0; k < 2; k++) {
    int flag; double stored; read_back(modules[k], flag, stored);
    std::fprintf(answer, "%s\"%s\":{\"objects_found\":%d,\"enabled\":%d,\"target_yaw_rad\":%.9f}", k ? "," : "",
                 modules[k].name, modules[k].found, flag, stored);
  }
  std::fputs("}}\n", answer);
  std::fclose(answer);
  std::rename(temporary.c_str(), (directory + "/heading_applied.json").c_str());
}

void* follow_target(void* argument) {
  const std::string directory = static_cast<const char*>(argument);
  const std::string source = directory + "/heading_target.bin";
  int input = -1, status = -1;
  bool located = false, failed = false;
  double last_sequence = -1., last_target = NAN, calls = 0.;
  int last_enable = -1;
  for (;; usleep(5000)) {
    if (failed) continue;
    if (input < 0 && (input = open(source.c_str(), O_RDONLY)) < 0) continue;
    double record[4];
    if (pread(input, record, sizeof record, 0) != static_cast<ssize_t>(sizeof record)) continue;
    const double sequence = record[0], target = record[2];
    const int enable = record[1] == 1. ? 1 : record[1] == 0. ? 0 : -1;
    // A record caught half written does not add up; the next poll reads it whole.
    if (sequence == last_sequence || enable < 0 || !std::isfinite(target) || record[3] != sequence + record[1] + target) continue;
    if (!located) {
      const long long scanned = locate();
      located = true;
      const char* error = scanned < 0 ? "process memory map not readable"
                          : (modules[0].found == 1 || modules[1].found == 1) ? nullptr : "no walking module identified";
      if (!error) { calls += apply(enable == 1, target) > 0; last_enable = enable; last_target = target; }
      report(directory, enable, target, scanned, error);
      failed = error != nullptr;
      if (failed) continue;
      status = open((directory + "/heading_status.bin").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    } else if (enable != last_enable || std::fabs(target - last_target) >= STEP) {
      calls += apply(enable == 1, target) > 0; last_enable = enable; last_target = target;
    }
    last_sequence = sequence;
    if (status >= 0) {
      int flag; double stored; read_back(modules[1], flag, stored);
      const double state[4] = {calls, sequence, static_cast<double>(flag), stored};
      if (pwrite(status, state, sizeof state, 0) != static_cast<ssize_t>(sizeof state)) { close(status); status = -1; }
    }
  }
  return nullptr;
}

__attribute__((constructor)) void start() {
  const char* directory = std::getenv("IGRIS_HEADING_DIR");
  if (!directory || !dlsym(RTLD_DEFAULT, modules[1].setter)) return;      // not the controller process
  pthread_t thread;
  if (pthread_create(&thread, nullptr, follow_target, const_cast<char*>(directory)) == 0) pthread_detach(thread);
}
}  // namespace
