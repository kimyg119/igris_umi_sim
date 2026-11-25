// Opt-in scheduler clock for one fingerprinted simulation executable.
// Four audited RobotHandler::inference call sites only. Control, DDS,
// SlowStart, SHM freshness, all other threads/call sites retain real time.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <ctime>

static bool policy_clock_callsite(std::uintptr_t offset) {
  return offset==0x5ab8a || offset==0x5ac64 || offset==0x5ace9 || offset==0x5ad00;
}

struct NativeTimeline {
  bool anchored=false;
  bool initialized=false;
  double last_sim=0.;
  std::int64_t value=0;
  std::int64_t sample(std::int64_t real, bool valid, double sim) {
    if(!anchored){value=real;anchored=true;}
    if(!initialized) {
      if(valid) {initialized=true;last_sim=sim;}
      return value;
    }
    if(valid) {
      // A simulator reset starts a new epoch, never reverses steady time.
      if(sim>=last_sim)value+=std::llround((sim-last_sim)*1e9);
      last_sim=sim;
    }
    // A paused/stale native stream cannot trigger extra policy ticks.
    // Independent wall-clock watchdogs are deliberately not modified.
    return value;
  }
};

struct Clock {std::uint64_t sequence;double wall,monotonic,sim;};
static_assert(sizeof(Clock)==32);

static bool snapshot(double& sim) {
  static const Clock* mapped=nullptr;
  if(!mapped) {
    const char* path=std::getenv("IGRIS_HLIK_CLOCK");if(!path)return false;
    int fd=open(path,O_RDONLY);if(fd<0)return false;
    struct stat st{};
    if(fstat(fd,&st)||st.st_size!=sizeof(Clock)){close(fd);return false;}
    void* p=mmap(nullptr,sizeof(Clock),PROT_READ,MAP_SHARED,fd,0);close(fd);
    if(p==MAP_FAILED)return false;mapped=static_cast<const Clock*>(p);
  }
  for(int attempt=0;attempt<8;attempt++) {
    auto before=__atomic_load_n(&mapped->sequence,__ATOMIC_SEQ_CST);
    if(!before || (before&1))continue;
    double monotonic=mapped->monotonic;double candidate=mapped->sim;
    auto after=__atomic_load_n(&mapped->sequence,__ATOMIC_SEQ_CST);
    timespec now{};clock_gettime(CLOCK_MONOTONIC,&now);
    double age=now.tv_sec+now.tv_nsec*1e-9-monotonic;
    if(before==after && std::isfinite(candidate) && candidate>=0. && age>=0. && age<.1) {
      sim=candidate;return true;
    }
  }
  return false;
}

static bool selected_process() {
  static const bool selected=[] {
    const char* expected=std::getenv("IGRIS_ROOM_POLICY_CLOCK_BINARY");
    struct stat a{},b{};
    return expected && stat(expected,&a)==0 && stat("/proc/self/exe",&b)==0
      && a.st_dev==b.st_dev && a.st_ino==b.st_ino;
  }();
  return selected;
}

#ifndef IGRIS_POLICY_CLOCK_TEST
extern "C" std::int64_t policy_steady_now() asm("_ZNSt6chrono3_V212steady_clock3nowEv");
extern "C" std::int64_t policy_steady_now() {
  static auto original=reinterpret_cast<std::int64_t(*)()>(
    dlsym(RTLD_NEXT,"_ZNSt6chrono3_V212steady_clock3nowEv"));
  if(!original)std::abort();
  const auto real=original();
  if(!selected_process())return real;
  Dl_info info{};void* caller=__builtin_return_address(0);
  if(!dladdr(caller,&info))return real;
  const auto offset=reinterpret_cast<std::uintptr_t>(caller)-reinterpret_cast<std::uintptr_t>(info.dli_fbase);
  if(!policy_clock_callsite(offset))return real;
  struct stat caller_file{},executable{};
  if(!info.dli_fname || stat(info.dli_fname,&caller_file) || stat("/proc/self/exe",&executable)
      || caller_file.st_dev!=executable.st_dev || caller_file.st_ino!=executable.st_ino)return real;
  thread_local NativeTimeline timeline;
  double sim=0.;const bool valid=snapshot(sim);
  const auto result=timeline.sample(real,valid,sim);
  // One trace per actual inference start, not per scheduler polling call.
  if(offset==0x5ace9) {
    thread_local FILE* log=nullptr;
    if(!log) {
      const char* output=std::getenv("IGRIS_ROOM_OUTPUT");
      if(!output)std::abort();
      char path[4096];
      if(std::snprintf(path,sizeof(path),"%s/vendor_policy_clock.jsonl",output)>=int(sizeof(path)))std::abort();
      log=std::fopen(path,"a");if(!log)std::abort();
      std::setvbuf(log,nullptr,_IOLBF,0);
    }
    std::fprintf(log,"{\"real_monotonic_ns\":%lld,\"scheduler_ns\":%lld,\"native_sim\":%.12g,\"native_fresh\":%s}\n",
                 static_cast<long long>(real),static_cast<long long>(result),sim,valid?"true":"false");
  }
  return result;
}
#endif
