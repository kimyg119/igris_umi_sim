// Read-only MuJoCo 3.2.7 observer. The vendor still owns every physics step.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/syscall.h>
#include "contact_wrench.h"
#include "support_force_groups.h"

static double wall() {
    timespec t; clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double thread_cpu() {
    timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return t.tv_sec+t.tv_nsec*1e-9;
}

// Isaac owns rendering. Suppress only the hidden native viewer's draw calls;
// do not bypass scene synchronization, input handling, or any dynamics call.
extern "C" void mjr_render(mjrRect viewport, mjvScene* scene, const mjrContext* context) {
    if(std::getenv("IGRIS_ROOM_SKIP_NATIVE_DRAW"))return;
    static auto draw=reinterpret_cast<void(*)(mjrRect,mjvScene*,const mjrContext*)>(dlsym(RTLD_NEXT,"mjr_render"));
    if(!draw)std::abort();draw(viewport,scene,context);
}
extern "C" void mjui_render(mjUI* ui, const mjuiState* state, const mjrContext* context) {
    if(std::getenv("IGRIS_ROOM_SKIP_NATIVE_DRAW"))return;
    static auto draw=reinterpret_cast<void(*)(mjUI*,const mjuiState*,const mjrContext*)>(dlsym(RTLD_NEXT,"mjui_render"));
    if(!draw)std::abort();draw(ui,state,context);
}

extern "C" void mj_step(const mjModel* m, mjData* d) {
    static auto step = reinterpret_cast<void(*)(const mjModel*, mjData*)>(dlsym(RTLD_NEXT,"mj_step"));
    if (!step) std::abort();
    const char* dir = std::getenv("IGRIS_ROOM_OUTPUT");
    const char* payload=std::getenv("IGRIS_ROOM_PAYLOAD");
    static SupportForceGroups support_groups;
    // Reject invalid selectors before the first observed vendor step.
    if(dir)support_groups.prepare(m,payload,dir);
    const double before=wall(),cpu_before=thread_cpu();
    step(m,d);
    if (!dir) return;
    static unsigned long steps=0;
    static double step_seconds=0,cpu_seconds=0,stats_at=0,first_wall=before;
    steps++;step_seconds+=wall()-before;cpu_seconds+=thread_cpu()-cpu_before;
    static double last = 0;
    double now = wall();
    if (now-last < 1.0/30.0) return;
    last = now;
    static const std::string base(dir);
    static FILE* log = std::fopen((base+"/poses.bin").c_str(),"wb");
    if (!log) { std::perror("pose log"); std::abort(); }
    std::vector<double> row{now,d->time};
    for (int i=0;i<m->nbody;i++) {
        row.insert(row.end(), d->xpos+3*i, d->xpos+3*i+3);
        row.insert(row.end(), d->xquat+4*i, d->xquat+4*i+4);
    }
    if (std::fwrite(row.data(),sizeof(double),row.size(),log)!=row.size() || std::fflush(log)) std::abort();
    FILE* latest=std::fopen((base+"/latest.tmp").c_str(),"wb");
    if (!latest || std::fwrite(row.data(),sizeof(double),row.size(),latest)!=row.size()) std::abort();
    if (std::fclose(latest) || std::rename((base+"/latest.tmp").c_str(),(base+"/latest.bin").c_str())) std::abort();
    static FILE* offsets=std::fopen((base+"/joint_offsets.bin").c_str(),"wb");
 static const char* offset_names[31]={"waist_yaw_backlash","waist_roll_backlash","waist_pitch_backlash","l_hip_pitch_backlash","l_hip_roll_backlash","l_hip_yaw_backlash","l_knee_pitch_backlash","l_ankle_pitch_backlash","l_ankle_roll_backlash","r_hip_pitch_backlash","r_hip_roll_backlash","r_hip_yaw_backlash","r_knee_pitch_backlash","r_ankle_pitch_backlash","r_ankle_roll_backlash","l_shoulder_pitch_backlash","l_shoulder_roll_backlash","l_shoulder_yaw_backlash","l_elbow_pitch_backlash","l_wrist_yaw_backlash","l_wrist_roll_backlash","l_wrist_pitch_backlash","r_shoulder_pitch_backlash","r_shoulder_roll_backlash","r_shoulder_yaw_backlash","r_elbow_pitch_backlash","r_wrist_yaw_backlash","r_wrist_roll_backlash","r_wrist_pitch_backlash","neck_yaw_backlash","neck_pitch_backlash"};
 double offset_row[32]={now};
 for(int j=0;j<31;j++){int id=mj_name2id(m,mjOBJ_JOINT,offset_names[j]);if(id>=0)offset_row[j+1]=d->qpos[m->jnt_qposadr[id]];}
 if(!offsets)std::abort();fwrite(offset_row,sizeof(double),32,offsets);fflush(offsets);

    static FILE* contacts=std::fopen((base+"/contacts.jsonl").c_str(),"w");
    static FILE* payload_contacts=std::fopen((base+"/payload_contacts.jsonl").c_str(),"w");
    double grasp_force[4]{},payload_support_z=0,source_support_z=0,destination_support_z=0;
    if(!contacts || !payload_contacts)std::abort();
    for(int c=0;c<d->ncon;c++) {
        const auto& ct=d->contact[c];
        auto name1=mj_id2name(m,mjOBJ_GEOM,ct.geom1),name2=mj_id2name(m,mjOBJ_GEOM,ct.geom2);
        if(!name1||!name2)continue;
        auto body1=mj_id2name(m,mjOBJ_BODY,m->geom_bodyid[ct.geom1]),body2=mj_id2name(m,mjOBJ_BODY,m->geom_bodyid[ct.geom2]);
        if(payload && *payload && ((body1 && !std::strcmp(payload,body1)) || (body2 && !std::strcmp(payload,body2)))) {
            mjtNum wrench[6];mj_contactForce(m,d,c,wrench);
            if(wrench[0]>0) {
                const int payload_body=body2 && !std::strcmp(payload,body2) ? m->geom_bodyid[ct.geom2] : m->geom_bodyid[ct.geom1];
                const auto measured=payload_contact_wrench(m,d,c,payload_body);
                const char* other=m->geom_bodyid[ct.geom1]==payload_body ? name2 : name1;
                if(std::strncmp(other,"l_hand",6) && std::strncmp(other,"r_hand",6))payload_support_z+=measured.force_world[2];
                const int other_body=m->geom_bodyid[ct.geom1]==payload_body ? m->geom_bodyid[ct.geom2] : m->geom_bodyid[ct.geom1];
                if(support_groups.source[other_body])source_support_z+=measured.force_world[2];
                if(support_groups.destination[other_body])destination_support_z+=measured.force_world[2];
                std::fprintf(payload_contacts,"{\"wall\":%.9f,\"sim\":%.9f,\"geoms\":[\"%s\",\"%s\"],\"normal_force_N\":%.9g,\"point_world_m\":[%.9g,%.9g,%.9g],\"normal_world\":[%.9g,%.9g,%.9g],\"force_on_payload_world_N\":[%.9g,%.9g,%.9g],\"torque_about_payload_com_world_Nm\":[%.9g,%.9g,%.9g],\"payload_relative_velocity_world_mps\":[%.9g,%.9g,%.9g],\"cone_utilization\":%.9g}\n",
                    now,d->time,name1,name2,wrench[0],ct.pos[0],ct.pos[1],ct.pos[2],ct.frame[0],ct.frame[1],ct.frame[2],
                    measured.force_world[0],measured.force_world[1],measured.force_world[2],
                    measured.torque_about_com_world[0],measured.torque_about_com_world[1],measured.torque_about_com_world[2],
                    measured.relative_velocity_world[0],measured.relative_velocity_world[1],measured.relative_velocity_world[2],measured.cone_utilization);
            }
        }
        bool room1=std::strncmp(name1,"room_geom_",10)==0,room2=std::strncmp(name2,"room_geom_",10)==0;
        if(room1==room2 || m->geom_bodyid[room1?ct.geom2:ct.geom1]==0)continue;
        mjtNum force[6];mj_contactForce(m,d,c,force);
        if(force[0]<=0)continue;
        auto body=mj_id2name(m,mjOBJ_BODY,m->geom_bodyid[room1?ct.geom1:ct.geom2]);
        int robot_geom=room1?ct.geom2:ct.geom1;
        const char* mesh=m->geom_type[robot_geom]==mjGEOM_MESH ? mj_id2name(m,mjOBJ_MESH,m->geom_dataid[robot_geom]) : nullptr;
        const char* patch=mesh && std::strstr(mesh,"___1_5") ? "palm" : mesh && std::strstr(mesh,"___1_4") ? "tip" : mesh && std::strstr(mesh,"___1_6") ? "mcp_plate" : "other";
        const char* robot_name=room1?name2:name1;
        if(payload && body && !std::strcmp(body,payload) &&
           (!std::strncmp(robot_name,"l_hand",6)||!std::strncmp(robot_name,"r_hand",6)) &&
           (!std::strcmp(patch,"palm")||!std::strcmp(patch,"tip")))
            grasp_force[2*(robot_name[0]=='r')+(!std::strcmp(patch,"tip"))]+=force[0];
        std::fprintf(contacts,"{\"wall\":%.9f,\"sim\":%.9f,\"robot_geom\":\"%s\",\"patch\":\"%s\",\"room_geom\":\"%s\",\"room_body\":\"%s\",\"normal_force_N\":%.9g,\"distance_m\":%.9g}\n",now,d->time,room1?name2:name1,patch,room1?name1:name2,body?body:"",force[0],ct.dist);
    }
    std::fflush(contacts);
    std::fflush(payload_contacts);
    FILE* grasp=std::fopen((base+"/grasp_force.tmp").c_str(),"w");
    if(!grasp)std::abort();
    std::fprintf(grasp,"{\"wall\":%.9f,\"sim\":%.9f,\"palm_tip_force_N\":[%.9g,%.9g,%.9g,%.9g],\"payload_support_force_world_z_N\":%.9g,\"source_support_force_world_z_N\":%.9g,\"destination_support_force_world_z_N\":%.9g,\"support_force_groups_configured\":%s}\n",now,d->time,grasp_force[0],grasp_force[1],grasp_force[2],grasp_force[3],payload_support_z,source_support_z,destination_support_z,support_groups.configured?"true":"false");
    if(std::fclose(grasp)||std::rename((base+"/grasp_force.tmp").c_str(),(base+"/grasp_force.json").c_str()))std::abort();
    if(now-stats_at>=1.) {
        stats_at=now;
        std::vector<mjtNum> state(mj_stateSize(m,mjSTATE_INTEGRATION));
        mj_getState(m,d,state.data(),mjSTATE_INTEGRATION);
        FILE* snapshot=std::fopen((base+"/state.tmp").c_str(),"wb");
        if(!snapshot || std::fwrite(state.data(),sizeof(mjtNum),state.size(),snapshot)!=state.size())std::abort();
        if(std::fclose(snapshot)||std::rename((base+"/state.tmp").c_str(),(base+"/state.bin").c_str()))std::abort();
        FILE* stats=std::fopen((base+"/physics_stats.tmp").c_str(),"w");if(!stats)std::abort();
        std::fprintf(stats,"{\"pid\":%d,\"tid\":%ld,\"steps\":%lu,\"mean_step_ms\":%.9g,\"mean_step_cpu_ms\":%.9g,\"elapsed_wall_s\":%.9g,\"integrated_sim_s\":%.9g,\"ncon\":%d,\"nefc\":%d,\"solver_iterations\":%d,\"jacobian\":%d,\"timestep\":%.12g,\"threadpool_bound\":%s,\"nq\":%d,\"nv\":%d,\"timer_callback_units_per_call\":{",getpid(),syscall(SYS_gettid),steps,1000*step_seconds/steps,1000*cpu_seconds/steps,now-first_wall,steps*m->opt.timestep,d->ncon,d->nefc,d->solver_niter[0],m->opt.jacobian,m->opt.timestep,d->threadpool?"true":"false",m->nq,m->nv);
        // Data symbols cannot be unresolved in an LD_PRELOAD library used by
        // the ROS launch parent too. Resolve timer labels only in MuJoCo.
        static auto timer_names=reinterpret_cast<const char* const*>(dlsym(RTLD_NEXT,"mjTIMERSTRING"));
        if(!timer_names)std::abort();
        for(int t=0;t<mjNTIMER;t++)std::fprintf(stats,"%s\"%s\":%.9g",t?",":"",timer_names[t],d->timer[t].number?d->timer[t].duration/d->timer[t].number:0.);
        std::fprintf(stats,"}}\n");
        if(std::fclose(stats)||std::rename((base+"/physics_stats.tmp").c_str(),(base+"/physics_stats.json").c_str()))std::abort();
    }
}

static void require(bool ok, const char* field) {
    if (!ok) { std::fprintf(stderr,"Vendor parity failed: %s\n",field); std::exit(1); }
}

// Compile as an executable as well to check the *actual vendor library*, before launch.
int main(int argc, char** argv) {
    if (argc!=4 && argc!=5 && argc!=6 && argc!=8) { std::fprintf(stderr,"usage: audit vendor.xml scene.xml model.json [state.bin | --noslip-iterations N | --impratio R]\n"); return 2; }
    require(std::strcmp(mj_versionString(),"3.2.7")==0,"MuJoCo version");
    char error[4096]{};
    auto a=mj_loadXML(argv[1],nullptr,error,sizeof(error));
    if (!a) { std::fprintf(stderr,"%s\n",error); return 1; }
    auto b=mj_loadXML(argv[2],nullptr,error,sizeof(error));
    if (!b) { std::fprintf(stderr,"%s\n",error); return 1; }
    require(b->nu==a->nu && b->neq==a->neq,"actuator/equality counts");
    int original_jacobian=a->opt.jacobian;
    require(b->opt.jacobian==a->opt.jacobian || b->opt.jacobian==mjJAC_SPARSE,"authorized dense/sparse representation only");
    a->opt.jacobian=b->opt.jacobian;
    int original_noslip=a->opt.noslip_iterations;
    double original_impratio=a->opt.impratio;
    bool declared_noslip=false,declared_impratio=false;
    for(int arg=4;argc!=5 && arg<argc;arg+=2) {
        char* end=nullptr;
        if(!std::strcmp(argv[arg],"--noslip-iterations")) {
            long requested=std::strtol(argv[arg+1],&end,10);
            require(!declared_noslip && end && !*end && requested>=1 && requested<=10 && b->opt.noslip_iterations==requested,"declared contact refinement only");
            a->opt.noslip_iterations=b->opt.noslip_iterations;declared_noslip=true;
        } else {
            require(!std::strcmp(argv[arg],"--impratio"),"explicit contact refinement argument");
            double requested=std::strtod(argv[arg+1],&end);
            require(!declared_impratio && end && !*end && std::isfinite(requested) && requested>=1 && requested<=1000 && b->opt.impratio==requested,"declared friction impedance only");
            a->opt.impratio=b->opt.impratio;declared_impratio=true;
        }
    }
    require(std::memcmp(&a->opt,&b->opt,sizeof(mjOption))==0,"manufacturer solver options except declared Jacobian/contact refinement");
    #define SAME(field,n) require(std::memcmp(a->field,b->field,sizeof(*a->field)*(n))==0,#field)
    SAME(body_mass,a->nbody); SAME(body_inertia,a->nbody*3);
    SAME(body_ipos,a->nbody*3); SAME(body_iquat,a->nbody*4);
    SAME(body_pos,a->nbody*3); SAME(body_quat,a->nbody*4);
    SAME(jnt_type,a->njnt); SAME(jnt_pos,a->njnt*3); SAME(jnt_axis,a->njnt*3); SAME(jnt_range,a->njnt*2);
    SAME(jnt_stiffness,a->njnt); SAME(qpos_spring,a->nq);
    SAME(dof_armature,a->nv); SAME(dof_damping,a->nv); SAME(dof_frictionloss,a->nv);
    SAME(actuator_gear,a->nu*6); SAME(actuator_gainprm,a->nu*mjNGAIN); SAME(actuator_biasprm,a->nu*mjNBIAS);
    SAME(actuator_ctrlrange,a->nu*2); SAME(actuator_forcerange,a->nu*2);
    SAME(eq_type,a->neq); SAME(eq_data,a->neq*mjNEQDATA); SAME(eq_solref,a->neq*mjNREF); SAME(eq_solimp,a->neq*mjNIMP);
    SAME(qpos0,a->nq);
    for(int g=0;g<a->ngeom;g++) {
        int body=a->geom_bodyid[g];
        int h=b->body_geomadr[body]+g-a->body_geomadr[body];
        #define GEOM(field,n) require(std::memcmp(a->field+(g)*(n),b->field+(h)*(n),sizeof(*a->field)*(n))==0,#field)
        GEOM(geom_type,1); GEOM(geom_size,3); GEOM(geom_pos,3); GEOM(geom_quat,4);
        GEOM(geom_friction,3); GEOM(geom_solref,mjNREF); GEOM(geom_solimp,mjNIMP);
        GEOM(geom_condim,1);
        const char* body_name=mj_id2name(a,mjOBJ_BODY,body);
        bool hand=false;
        for(int ancestor=body;ancestor;ancestor=a->body_parentid[ancestor]) {
            auto name=mj_id2name(a,mjOBJ_BODY,ancestor);
            if(name && (!std::strcmp(name,"l_hand_rcv") || !std::strcmp(name,"r_hand_rcv")))hand=true;
        }
        if(hand) {
            if(!(a->geom_contype[g]==0 && a->geom_conaffinity[g]==0 && b->geom_contype[h]==2 && b->geom_conaffinity[h]==0))std::fprintf(stderr,"hand mask: %s geom=%s %d/%d -> %d/%d\n",body_name,mj_id2name(a,mjOBJ_GEOM,g),a->geom_contype[g],a->geom_conaffinity[g],b->geom_contype[h],b->geom_conaffinity[h]);
            require(a->geom_contype[g]==0 && a->geom_conaffinity[g]==0 && b->geom_contype[h]==2 && b->geom_conaffinity[h]==0,"authorized hand environment-only contact");
        } else {GEOM(geom_contype,1);GEOM(geom_conaffinity,1);}
    }
    // Confirm every original robot/ground collision-filter relationship remains
    // identical, including hands against the robot and each other.
    for(int g=0;g<a->ngeom;g++)for(int h=0;h<a->ngeom;h++) {
        auto map=[&](int k){int body=a->geom_bodyid[k];return b->body_geomadr[body]+k-a->body_geomadr[body];};
        int x=map(g),y=map(h);
        bool before=(a->geom_contype[g]&a->geom_conaffinity[h]) || (a->geom_contype[h]&a->geom_conaffinity[g]);
        bool after=(b->geom_contype[x]&b->geom_conaffinity[y]) || (b->geom_contype[y]&b->geom_conaffinity[x]);
        require(before==after,"original robot and ground collision eligibility");
    }
    for(int k=0;k<a->nkey;k++) {
        require(std::memcmp(a->key_qpos+k*a->nq,b->key_qpos+k*b->nq,sizeof(mjtNum)*a->nq)==0,"robot keyframes");
    }
    require(b->nq==a->nq+39*7 && b->nv==a->nv+39*6,"39 room free bodies");
    int boxes=0;
    for(int i=0;i<b->nbody;i++) {
        const char* name=mj_id2name(b,mjOBJ_BODY,i);
        if(!name || std::strncmp(name,"room_Boxes__",12))continue;
        boxes++;int g=b->body_geomadr[i];
        require(b->body_geomnum[i]==1 && b->geom_type[g]==mjGEOM_BOX && b->geom_contype[g]==1 && b->geom_conaffinity[g]==3,"every box must contact robot and hands");
    }
    require(boxes==38,"38 individually collidable boxes");
    auto d=mj_makeData(b); mj_forward(b,d);
    bool compliant=mj_name2id(b,mjOBJ_JOINT,"l_hand_mcp_compliant")>=0;
    if(compliant) {
        for(const char* side:{"l","r"}) {
            std::string prefix=std::string(side)+"_hand_";
            int mcp=mj_name2id(b,mjOBJ_JOINT,(prefix+"mcp_compliant").c_str());
            int dip=mj_name2id(b,mjOBJ_JOINT,(prefix+"dip_spring").c_str());
            require(mcp>=0 && dip>=0,"MCP and DIP joints");
            double mass=0;for(const char* part:{"rcv","mcp","dip"})mass+=b->body_mass[mj_name2id(b,mjOBJ_BODY,(prefix+part).c_str())];
            require(std::abs(mass-.6)<1e-10,"0.6 kg total hand mass");
            require(!b->jnt_limited[dip] && std::abs(b->jnt_stiffness[dip]-.04901587301587303)<1e-12,"passive torsion spring without artificial stops");
            d->qpos[b->jnt_qposadr[dip]]=.2;mj_forward(b,d);
            require(std::abs(d->qfrc_passive[b->jnt_dofadr[dip]]+.04901587301587303*.2)<1e-10,"native DIP spring torque");
            d->qpos[b->jnt_qposadr[dip]]=0;
        }
        mj_forward(b,d);
    }
    if(argc==5) {
        std::vector<mjtNum> state(mj_stateSize(b,mjSTATE_INTEGRATION));FILE* saved=std::fopen(argv[4],"rb");
        require(saved && std::fread(state.data(),sizeof(mjtNum),state.size(),saved)==state.size(),"complete native state snapshot");std::fclose(saved);
        mj_setState(b,d,state.data(),mjSTATE_INTEGRATION);mj_forward(b,d);
    }
    FILE* f=std::fopen(argv[3],"w"); if (!f) return 1;
    std::fprintf(f,"{\"vendor_robot_parameters_pass\":true,\"noslip_iterations\":%d,\"vendor_noslip_iterations\":%d,\"impratio\":%.17g,\"vendor_impratio\":%.17g,\"contact_solver_changed\":%s,\"authorized_compliant_hands\":%s,\"native_dip_spring_check\":%s,\"jacobian_changed\":%s,\"mujoco_version\":\"%s\",\"timestep\":%.17g,\"nq\":%d,\"nv\":%d,\"nu\":%d,\"neq\":%d,\"bodies\":[",b->opt.noslip_iterations,original_noslip,b->opt.impratio,original_impratio,b->opt.noslip_iterations==original_noslip && b->opt.impratio==original_impratio?"false":"true",compliant?"true":"false",compliant?"true":"null",b->opt.jacobian==original_jacobian?"false":"true",mj_versionString(),b->opt.timestep,b->nq,b->nv,b->nu,b->neq);
    for(int i=0;i<b->nbody;i++) {
        auto name=mj_id2name(b,mjOBJ_BODY,i);
        std::fprintf(f,"%s{\"name\":\"%s\",\"pose\":[",i?",":"",name?name:"");
        for(int k=0;k<7;k++) std::fprintf(f,"%s%.17g",k?",":"",k<3?d->xpos[3*i+k]:d->xquat[4*i+k-3]);
        std::fprintf(f,"]}");
    }
    std::fprintf(f,"],\"hinge_q\":{");bool comma=false;
    for(int j=0;j<b->njnt;j++)if(b->jnt_type[j]==mjJNT_HINGE) {
        const char* name=mj_id2name(b,mjOBJ_JOINT,j);if(!name)continue;
        std::fprintf(f,"%s\"%s\":%.17g",comma?",":"",name,d->qpos[b->jnt_qposadr[j]]);comma=true;
    }
    std::fprintf(f,"}}\n"); std::fclose(f);
    std::printf("Vendor 3.2.7 robot parameters PASS: nq=%d nv=%d nu=%d neq=%d; 39 dynamic room bodies; Jacobian changed=%s\n",b->nq,b->nv,b->nu,b->neq,b->opt.jacobian==original_jacobian?"false":"true");
    mj_deleteData(d); mj_deleteModel(a); mj_deleteModel(b);
}
