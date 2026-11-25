// SIM-only observation at the documented full RobotCommand mapping overload.
// No JointManager layout access, command mutation, extra mapping call, or physics call.
#include "igris_c_lib/kinematics/JointManager.h"
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

namespace {
using JM = robros::controller::igris_c::JointManager;
using RC = robros::shm::command::RobotCommand;
using MS = robros::shm::state::motorStatusData_st;
using CM = std::array<ParallelLinkCommandMode, 5>;
using IM = std::array<ParallelLinkIKSolverMode, 5>;
using SM = std::array<ParallelLinkSafetyMode, 5>;
using SS = std::array<ParallelLinkSoftSafeSetMode, 5>;
using BB = std::array<bool, 5>;
using Fn = int (*)(JM*, const MS&, const RC&, RC&, const CM&, const IM&,
                  const SM&, const SS&, const BB&, const BB&);
constexpr char SYMBOL[] = "_ZN6robros10controller7igris_c12JointManager18jointCommand2MotorERKNS_3shm5state18motorStatusData_stERKNS3_7command12RobotCommandERS9_RKSt5arrayI23ParallelLinkCommandModeLm5EERKSD_I24ParallelLinkIKSolverModeLm5EERKSD_I22ParallelLinkSafetyModeLm5EERKSD_I27ParallelLinkSoftSafeSetModeLm5EERKSD_IbLm5EESW_";
static_assert(MODEL_DOF == 31 && kParallelLinkModeCount == 5);
// This assertion resolves the overload against the actual installed declaration.
using Member = int (JM::*)(const MS&, const RC&, RC&, const CM&, const IM&,
                          const SM&, const SS&, const BB&, const BB&);
static_assert(std::is_same_v<decltype(static_cast<Member>(&JM::jointCommand2Motor)), Member>);

#pragma pack(push, 1)
struct Header { uint32_t seq, sec, nanosec; char frame_id[256]; };
struct Command {
  Header header;
  uint32_t kinematic_mode[5], control_mode[31];
  double q[31], dq[31], tau[31], kp[31], kd[31];
};
struct Motor {
  Header header;
  uint32_t motor_id[31];
  char joint_name[31][30], motor_name[31][30];
  double target_position[31], target_torque[31], current_position[31], current_velocity[31];
  double current_acceleration[31], current_torque[31], current_current[31], current_temp[31];
  uint32_t error_state[31];
  uint64_t packet_rate[31];
};
struct Record {
  uint64_t sequence, entry_monotonic_ns, exit_monotonic_ns;
  int32_t return_code;
  uint32_t reserved;
  int32_t options[6][5];
  Command before, after;
  Motor motor;
};
struct FileHeader {
  char magic[8];
  uint32_t version, header_bytes, record_bytes, capacity;
  uint64_t reserved_slots, dropped_capacity, dropped_busy, armed_monotonic_ns, pid;
};
#pragma pack(pop)
static_assert(sizeof(Header)==268 && sizeof(Command)==1652 && sizeof(Motor)==4608);
static_assert(sizeof(Record)==8064 && sizeof(FileHeader)==64);

uint64_t monotonic_ns() noexcept {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec)*1000000000ULL + uint64_t(ts.tv_nsec);
}

void capture_header(Header& dst, const robros::shm::common::Header& src) noexcept {
  dst.seq=src.seq; dst.sec=src.sec; dst.nanosec=src.nanosec;
  std::memcpy(dst.frame_id,src.frame_id,sizeof(dst.frame_id));
}
void capture(Command& dst, const RC& src) noexcept {
  capture_header(dst.header,src.header);
  for(int i=0;i<5;++i) dst.kinematic_mode[i]=uint32_t(src.kinematicModes[i]);
  for(int i=0;i<31;++i) {
    dst.control_mode[i]=uint32_t(src.controlMode[i]);
    dst.q[i]=src.q[i]; dst.dq[i]=src.qDot[i]; dst.tau[i]=src.torque[i];
    dst.kp[i]=src.kp[i]; dst.kd[i]=src.kd[i];
  }
}
void capture(Motor& dst, const MS& src) noexcept {
  capture_header(dst.header,src.header);
#define COPY_FIELD(field) std::memcpy(dst.field,src.field,sizeof(dst.field))
  COPY_FIELD(motor_id); COPY_FIELD(joint_name); COPY_FIELD(motor_name);
  COPY_FIELD(target_position); COPY_FIELD(target_torque); COPY_FIELD(current_position);
  COPY_FIELD(current_velocity); COPY_FIELD(current_acceleration); COPY_FIELD(current_torque);
  COPY_FIELD(current_current); COPY_FIELD(current_temp); COPY_FIELD(error_state); COPY_FIELD(packet_rate);
#undef COPY_FIELD
}

struct Trace {
  std::atomic<FileHeader*> header{nullptr};
  std::atomic<bool> disabled{false};
  std::mutex initialization;
  const char* arm = std::getenv("IGRIS_HTD_COMMAND_OBSERVER_ARM");
  const char* path = std::getenv("IGRIS_HTD_COMMAND_OBSERVER_TRACE");
  const char* expected = std::getenv("IGRIS_HTD_COMMAND_OBSERVER_EXE");

  Trace() noexcept {
    char actual[PATH_MAX]{};
    const auto n=readlink("/proc/self/exe",actual,sizeof(actual)-1);
    if(n<=0 || !arm || !path || !expected || arm[0]!='/' || path[0]!='/' ||
       std::strcmp(actual,expected)!=0) disabled=true;
  }
  FileHeader* get() noexcept {
    auto* ready=header.load(std::memory_order_acquire);
    if(ready || disabled.load(std::memory_order_relaxed)) return ready;
    if(access(arm,F_OK)!=0) return nullptr;
    // Never wait for another observer invocation. Normal SIM controller has one caller.
    std::unique_lock<std::mutex> lock(initialization,std::try_to_lock);
    if(!lock.owns_lock()) return nullptr;
    ready=header.load(std::memory_order_acquire);
    if(ready || disabled.load()) return ready;
    disabled=true;  // Any arming failure affects logging only; never retry/truncate evidence.
    const char* text=std::getenv("IGRIS_HTD_COMMAND_OBSERVER_MAX_RECORDS");
    char* end=nullptr;
    const long count=text?std::strtol(text,&end,10):4096;
    if(count<1 || count>8192 || (text && (!end || *end))) return nullptr;
    const size_t bytes=sizeof(FileHeader)+size_t(count)*sizeof(Record);
    const int fd=open(path,O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd<0) return nullptr;
    if(ftruncate(fd,off_t(bytes))!=0) { close(fd); return nullptr; }
    void* mem=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    close(fd);
    if(mem==MAP_FAILED) return nullptr;
    ready=static_cast<FileHeader*>(mem);
    // New file is zero-filled. Touch pages before timing-sensitive mapping begins.
    for(size_t i=0;i<bytes;i+=4096) static_cast<volatile char*>(mem)[i]=0;
    std::memcpy(ready->magic,"HTDCMD01",8);
    ready->version=1; ready->header_bytes=sizeof(FileHeader); ready->record_bytes=sizeof(Record);
    ready->capacity=uint32_t(count); ready->armed_monotonic_ns=monotonic_ns(); ready->pid=getpid();
    header.store(ready,std::memory_order_release);
    return ready;
  }
  Record* reserve() noexcept {
    auto* h=get();
    if(!h) return nullptr;
    const uint64_t index=__atomic_fetch_add(&h->reserved_slots,1,__ATOMIC_RELAXED);
    if(index>=h->capacity) {
      __atomic_fetch_add(&h->dropped_capacity,1,__ATOMIC_RELAXED);
      return nullptr;
    }
    return reinterpret_cast<Record*>(reinterpret_cast<char*>(h)+sizeof(FileHeader))+index;
  }
};
Trace& trace() { static Trace instance; return instance; }
Fn original() {
  static Fn fn=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,SYMBOL));
  // An absent underlying symbol is a loader/configuration failure. Never fabricate a successful mapping.
  if(!fn) std::abort();
  return fn;
}
} // namespace

namespace robros::controller::igris_c {
__attribute__((visibility("default")))
int JointManager::jointCommand2Motor(const MS& motor, const RC& from, RC& to,
    const CM& cm, const IM& im, const SM& sm, const SS& ss, const BB& gains, const BB& ff) {
  const int entry_errno=errno;
  Fn fn=original();
  Record* rec=trace().reserve();
  if(rec) {
    capture(rec->before,from); capture(rec->motor,motor);
    for(int i=0;i<5;++i) {
      rec->options[0][i]=int(cm[i]); rec->options[1][i]=int(im[i]);
      rec->options[2][i]=int(sm[i]); rec->options[3][i]=int(ss[i]);
      rec->options[4][i]=int(gains[i]); rec->options[5][i]=int(ff[i]);
    }
    rec->entry_monotonic_ns=monotonic_ns();
  }
  errno=entry_errno;
  const int result=fn(this,motor,from,to,cm,im,sm,ss,gains,ff);
  const int result_errno=errno;
  if(rec) {
    rec->exit_monotonic_ns=monotonic_ns();
    rec->return_code=result; capture(rec->after,to);
    auto* h=trace().header.load(std::memory_order_acquire);
    const auto* first=reinterpret_cast<Record*>(reinterpret_cast<char*>(h)+sizeof(FileHeader));
    __atomic_store_n(&rec->sequence,uint64_t(rec-first)+1,__ATOMIC_RELEASE);
  }
  errno=result_errno;
  return result;
}
} // namespace robros::controller::igris_c
