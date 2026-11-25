// Contact coverage and timing against the installed, unmodified vendor library.
// The contact probe relocates one box in an isolated mjData; it sends no commands.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double clock_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc,char** argv) {
    if(argc!=5) {std::fprintf(stderr,"usage: check_physics scene.xml result.json steps threads\n");return 2;}
    char error[4096]{};
    auto m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
    if(!m) {std::fprintf(stderr,"%s\n",error);return 1;}
    int steps=std::atoi(argv[3]),threads=std::atoi(argv[4]);
    if(steps<1 || threads<0 || threads>8 || std::strcmp(mj_versionString(),"3.2.7"))return 2;
    auto d=mj_makeData(m);
    auto pool=threads?mju_threadPoolCreate(threads):nullptr;
    if(pool)mju_bindThreadPool(d,pool);
    FILE* f=std::fopen(argv[2],"w");if(!f)return 1;
    std::fprintf(f,"{\"mujoco_version\":\"%s\",\"physics_options_modified\":false,\"jacobian\":%d,\"threads\":%d,\"contact_probes\":[",mj_versionString(),m->opt.jacobian,threads);
    int boxes=0,passed=0,tests=0;
    for(int b=1;b<m->nbody;b++) {
        const char* name=mj_id2name(m,mjOBJ_BODY,b);
        if(!name || std::strncmp(name,"room_Boxes__",12))continue;
        boxes++;
        int box=m->body_geomadr[b];
        for(const char* target:{"l_forearm_collision","l_lower_leg_collision","l_hand_rcv_g0","r_hand_rcv_g0"}) {
            int g=mj_name2id(m,mjOBJ_GEOM,target);
            if(g<0 || m->body_geomnum[b]!=1 || m->jnt_type[m->body_jntadr[b]]!=mjJNT_FREE)return 1;
            mj_resetDataKeyframe(m,d,0);mj_forward(m,d);
            int q=m->jnt_qposadr[m->body_jntadr[b]];
            for(int k=0;k<3;k++)d->qpos[q+k]=d->geom_xpos[3*g+k];
            d->qpos[q+3]=1;d->qpos[q+4]=d->qpos[q+5]=d->qpos[q+6]=0;
            mj_forward(m,d);
            int contacts=0;double force=0,min_dist=0;
            for(int c=0;c<d->ncon;c++) {
                const auto& contact=d->contact[c];
                if((contact.geom1==g && contact.geom2==box)||(contact.geom1==box && contact.geom2==g)) {
                    contacts++;min_dist=std::min(min_dist,contact.dist);
                    mjtNum wrench[6];mj_contactForce(m,d,c,wrench);force+=std::abs(wrench[0]);
                }
            }
            bool ok=contacts>0 && min_dist<0 && force>1e-6;
            std::fprintf(f,"%s{\"box\":\"%s\",\"robot_geom\":\"%s\",\"contacts\":%d,\"normal_force_N\":%.17g,\"penetration_m\":%.17g,\"pass\":%s}",tests?",":"",name,target,contacts,force,-min_dist,ok?"true":"false");
            tests++;passed+=ok;
        }
    }
    std::fprintf(f,"],\"boxes\":%d,\"contact_tests\":%d,\"contact_tests_passed\":%d",boxes,tests,passed);
    mj_resetDataKeyframe(m,d,0);
    for(int i=0;i<100;i++)mj_step(m,d);
    std::memset(d->timer,0,sizeof(d->timer));mjcb_time=clock_seconds;
    std::vector<double> times;times.reserve(steps);
    double contacts=0,constraints=0;double start=clock_seconds();
    for(int i=0;i<steps;i++) {
        double before=clock_seconds();mj_step(m,d);times.push_back(clock_seconds()-before);
        contacts+=d->ncon;constraints+=d->nefc;
    }
    double elapsed=clock_seconds()-start;std::sort(times.begin(),times.end());
    std::fprintf(f,",\"steps\":%d,\"seconds\":%.17g,\"mean_step_ms\":%.17g,\"p95_step_ms\":%.17g,\"max_step_ms\":%.17g,\"required_step_ms\":%.17g,\"compute_realtime_factor\":%.17g,\"mean_contacts\":%.17g,\"mean_constraints\":%.17g,\"timers_ms_per_step\":{",steps,elapsed,1000*elapsed/steps,1000*times[int(.95*(steps-1))],1000*times.back(),1000*m->opt.timestep,steps*m->opt.timestep/elapsed,contacts/steps,constraints/steps);
    for(int i=0;i<mjNTIMER;i++)std::fprintf(f,"%s\"%s\":%.17g",i?",":"",mjTIMERSTRING[i],1000*d->timer[i].duration/steps);
    std::fprintf(f,"},\"contact_coverage_pass\":%s}\n",boxes==38&&tests==passed?"true":"false");std::fclose(f);
    std::printf("boxes=%d contacts=%d/%d mean_step_ms=%.4f compute_RTF=%.4f\n",boxes,passed,tests,1000*elapsed/steps,steps*m->opt.timestep/elapsed);
    mj_deleteData(d);if(pool)mju_threadPoolDestroy(pool);mj_deleteModel(m);
    return boxes==38&&passed==tests?0:1;
}
