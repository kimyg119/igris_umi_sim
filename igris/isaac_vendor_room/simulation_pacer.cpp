// Optional wall-time pacing only. The vendor retains every physics/control step.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <unistd.h>

static double monotonic_seconds() {
    timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}

static double configured_rate(const char* variable="IGRIS_ROOM_MAX_REALTIME_FACTOR") {
    const char* value=std::getenv(variable);
    if(!value)return 0.;
    char* end=nullptr;double rate=std::strtod(value,&end);
    if(end==value || *end || !std::isfinite(rate) || rate<=0 || rate>1) {
        std::fprintf(stderr,"Invalid %s; expected (0,1]\n",variable);
        std::exit(2);
    }
    return rate;
}

static double slowdown_rate(const char* path,double initial) {
    if(!path)return 0.;
    double rate=configured_rate("IGRIS_ROOM_SLOW_REALTIME_FACTOR");
    if(!rate || !initial || rate>initial) {
        std::fprintf(stderr,"Scoped slowdown requires a positive rate no greater than the initial rate\n");
        std::exit(2);
    }
    return rate;
}

static void acknowledge_slowdown(const char* path,double previous,double rate,double sim) {
    std::string destination=std::string(path)+".ack",temporary=destination+".tmp";
    FILE* file=std::fopen(temporary.c_str(),"w");
    if(!file)std::abort();
    if(std::fprintf(file,"{\"previous_rate\":%.17g,\"rate\":%.17g,\"native_time_s\":%.17g,\"monotonic_s\":%.17g}\n",
            previous,rate,sim,monotonic_seconds())<0)std::abort();
    if(std::fclose(file) || std::rename(temporary.c_str(),destination.c_str()))std::abort();
}

extern "C" void mj_step(const mjModel* m,mjData* d) {
    static double rate=configured_rate();
    static const char* start_path=std::getenv("IGRIS_ROOM_PACER_START");
    static const char* slow_path=std::getenv("IGRIS_ROOM_PACER_SLOWDOWN");
    static const double slow_rate=slowdown_rate(slow_path,rate);
    static bool slowed=false;
    static bool activated=start_path==nullptr;
    static auto step=reinterpret_cast<void(*)(const mjModel*,mjData*)>(dlsym(RTLD_NEXT,"mj_step"));
    if(!step)std::abort();
    if(rate && !activated)activated=access(start_path,F_OK)==0;
    if(rate && activated) {
        if(slow_path && !slowed && access(slow_path,F_OK)==0) {
            double previous=rate;rate=slow_rate;slowed=true;
            acknowledge_slowdown(slow_path,previous,rate,d->time);
        }
        static thread_local double deadline=0.;
        double now=monotonic_seconds();
        if(deadline>now) {
            timespec target{static_cast<time_t>(deadline),0};
            target.tv_nsec=static_cast<long>((deadline-target.tv_sec)*1e9);
            int status;
            do {status=clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&target,nullptr);} while(status==EINTR);
            if(status)std::abort();
            now=monotonic_seconds();
        }
        // Missed deadlines never cause a catch-up burst of simulation steps.
        deadline=now+m->opt.timestep/rate;
    }
    step(m,d);
}
