// Latest native clock, written every physics step, independent of 30 Hz logs.
// No physics state or control is modified. Linux shared mmap uses a seqlock.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <ctime>

struct Clock { std::uint64_t sequence; double wall, monotonic, sim; };
static double seconds(clockid_t id) {
  timespec t; clock_gettime(id,&t); return t.tv_sec+t.tv_nsec*1e-9;
}
extern "C" void mj_step(const mjModel* m,mjData* d) {
  static auto step=(void(*)(const mjModel*,mjData*))dlsym(RTLD_NEXT,"mj_step");
  step(m,d);
  static Clock* clock=nullptr;
  if(!clock) {
    const char* path=getenv("IGRIS_HLIK_CLOCK"); if(!path)return;
    int fd=open(path,O_RDWR|O_CREAT|O_TRUNC,0600); if(fd<0)return;
    if(ftruncate(fd,sizeof(Clock))) {close(fd);return;}
    void* memory=mmap(nullptr,sizeof(Clock),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    close(fd);if(memory==MAP_FAILED)return;clock=static_cast<Clock*>(memory);
    __atomic_store_n(&clock->sequence,0,__ATOMIC_SEQ_CST);
  }
  auto seq=__atomic_load_n(&clock->sequence,__ATOMIC_SEQ_CST);
  __atomic_store_n(&clock->sequence,seq+1,__ATOMIC_SEQ_CST);
  clock->wall=seconds(CLOCK_REALTIME);clock->monotonic=seconds(CLOCK_MONOTONIC);clock->sim=d->time;
  __atomic_store_n(&clock->sequence,seq+2,__ATOMIC_SEQ_CST);
}
