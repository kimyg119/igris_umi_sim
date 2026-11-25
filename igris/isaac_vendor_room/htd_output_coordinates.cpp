// Read-only native observer. The vendor owns all dynamics and FK calls.
#include <mujoco/mujoco.h>
#include "igris_c_shm/shm/state.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <ctime>

using Motor = robros::shm::state::motorStatusData_st;
using Joint = robros::shm::state::jointStatusData_st;
// Same public declaration/signature as the supplied JointManager.h. Its return
// value and every argument are forwarded unchanged; no second FK is executed.
namespace robros::controller::igris_c {
class JointManager { public: int motor2Joint(const Motor&, Joint&); };
}
struct Snapshot {
  std::uint64_t sequence, version;
  double wall, monotonic, sim;
  // hinge q/dq, backlash q/dq, output q/dq, encoder serial q/dq, motor q/dq
  double v[10][31];
};
static_assert(sizeof(Snapshot)==2520);
static_assert(offsetof(Motor,current_position)==0xac0);
static_assert(offsetof(Motor,current_velocity)==0xbb8);
static_assert(offsetof(Joint,current_position)==0x798);
static_assert(offsetof(Joint,current_velocity)==0x890);
static thread_local const mjModel* active_model=nullptr;
static thread_local mjData* active_data=nullptr;
static void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
static double seconds(clockid_t clock) {timespec t;clock_gettime(clock,&t);return t.tv_sec+t.tv_nsec*1e-9;}
struct Mapping {
  int hinge[31],backlash[31],motor[31];
  explicit Mapping(const mjModel* m) {
    require(m->nu>=31,"missing native motor actuators");
    const char* motors[31]={"waist_yaw","l_waist_flange","r_waist_flange","l_hip_pitch","l_hip_roll","l_hip_yaw","l_knee_pitch","l_ankle_flange_out","l_ankle_flange_in","r_hip_pitch","r_hip_roll","r_hip_yaw","r_knee_pitch","r_ankle_flange_out","r_ankle_flange_in","l_shoulder_pitch","l_shoulder_roll","l_shoulder_yaw","l_elbow_pitch","l_wrist_yaw","l_wrist_motor_output_front","l_wrist_motor_output_back","r_shoulder_pitch","r_shoulder_roll","r_shoulder_yaw","r_elbow_pitch","r_wrist_yaw","r_wrist_motor_output_front","r_wrist_motor_output_back","neck_yaw","neck_pitch"};
    for(int i=0;i<31;i++) {
      hinge[i]=mj_name2id(m,mjOBJ_JOINT,joint_name[i]);
      backlash[i]=mj_name2id(m,mjOBJ_JOINT,(std::string(joint_name[i])+"_backlash").c_str());
      require(hinge[i]>=0 && m->jnt_type[hinge[i]]==mjJNT_HINGE,"missing serial hinge");
      require((i<29)==(backlash[i]>=0),"unexpected backlash joint map");
      if(backlash[i]>=0) {
        int h=hinge[i],b=backlash[i];
        require(m->jnt_type[b]==mjJNT_HINGE && m->jnt_bodyid[h]==m->jnt_bodyid[b],"backlash parent/type mismatch");
        for(int k=0;k<3;k++) require(std::abs(m->jnt_axis[3*h+k]-m->jnt_axis[3*b+k])<1e-12 && std::abs(m->jnt_pos[3*h+k]-m->jnt_pos[3*b+k])<1e-12,"backlash axis/origin mismatch");
      }
      require(mj_name2id(m,mjOBJ_ACTUATOR,motors[i])==i,"native motor SDK ordering mismatch");
      motor[i]=m->actuator_trnid[2*i];
      require(motor[i]>=0 && m->jnt_type[motor[i]]==mjJNT_HINGE,"native motor hinge mismatch");
    }
  }
  void extract(const mjModel* m,const mjData* d,Snapshot& s) const {
    for(int i=0;i<31;i++) {
      s.v[0][i]=d->qpos[m->jnt_qposadr[hinge[i]]];
      s.v[1][i]=d->qvel[m->jnt_dofadr[hinge[i]]];
      s.v[2][i]=backlash[i]<0?0:d->qpos[m->jnt_qposadr[backlash[i]]];
      s.v[3][i]=backlash[i]<0?0:d->qvel[m->jnt_dofadr[backlash[i]]];
      s.v[4][i]=s.v[0][i]+s.v[2][i];s.v[5][i]=s.v[1][i]+s.v[3][i];
      s.v[8][i]=d->qpos[m->jnt_qposadr[motor[i]]];
      s.v[9][i]=d->qvel[m->jnt_dofadr[motor[i]]];
    }
  }
};
extern "C" void mj_step(const mjModel* m,mjData* d) {
  static auto next=reinterpret_cast<void(*)(const mjModel*,mjData*)>(dlsym(RTLD_NEXT,"mj_step"));
  if(!next)std::abort();
  active_model=m;active_data=d;
  next(m,d);
  active_model=nullptr;active_data=nullptr;
}
int robros::controller::igris_c::JointManager::motor2Joint(const Motor& from,Joint& to) {
  using F=int(*)(void*,const Motor&,Joint&);
  static auto next=reinterpret_cast<F>(dlsym(RTLD_NEXT,"_ZN6robros10controller7igris_c12JointManager11motor2JointERKNS_3shm5state18motorStatusData_stERNS4_18jointStatusData_stE"));
  if(!next)std::abort();
  int result=next(this,from,to);
  if(!active_model || !active_data)return result;
  const char* directory=std::getenv("IGRIS_ROOM_OUTPUT");if(!directory)return result;
  try {
    static Mapping map(active_model);
    static Snapshot* memory=nullptr;
    static FILE* log=nullptr;
    if(!memory) {
      std::string base(directory);
      int fd=open((base+"/output_coordinates.bin").c_str(),O_RDWR|O_CREAT|O_TRUNC,0600);
      require(fd>=0,"open output coordinates");
      require(ftruncate(fd,sizeof(Snapshot))==0,"size output coordinates");
      void* p=mmap(nullptr,sizeof(Snapshot),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);
      require(p!=MAP_FAILED,"mmap output coordinates");memory=static_cast<Snapshot*>(p);
      std::memset(memory,0,sizeof(*memory));
      log=std::fopen((base+"/output_coordinates_trace.bin").c_str(),"wb");require(log,"open coordinate trace");
    }
    Snapshot s{};s.version=1;s.wall=seconds(CLOCK_REALTIME);s.monotonic=seconds(CLOCK_MONOTONIC);s.sim=active_data->time;
    map.extract(active_model,active_data,s);
    for(int i=0;i<31;i++) {
      // This proves the FK result and native hinges belong to the SAME state.
      require(std::abs(s.v[8][i]-from.current_position[i])<1e-9 && std::abs(s.v[9][i]-from.current_velocity[i])<1e-9,"native/FK timestamp or motor mapping mismatch");
      s.v[6][i]=to.current_position[i];s.v[7][i]=to.current_velocity[i];
      for(int k=0;k<10;k++)require(std::isfinite(s.v[k][i]),"nonfinite native coordinate");
    }
    auto sequence=__atomic_load_n(&memory->sequence,__ATOMIC_SEQ_CST);
    __atomic_store_n(&memory->sequence,sequence+1,__ATOMIC_SEQ_CST);
    std::memcpy(reinterpret_cast<char*>(memory)+8,reinterpret_cast<char*>(&s)+8,sizeof(s)-8);
    __atomic_store_n(&memory->sequence,sequence+2,__ATOMIC_SEQ_CST);
    s.sequence=sequence+2;
    require(std::fwrite(&s,sizeof(s),1,log)==1 && std::fflush(log)==0,"write coordinate trace");
  } catch(const std::exception& e) {std::fprintf(stderr,"HTD_OUTPUT_OBSERVER: %s\n",e.what());std::abort();}
  return result;
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"usage: coordinate audit vendor.xml");
    char error[4096];mjModel* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));require(m,error);
    mjData* d=mj_makeData(m);Mapping map(m);Snapshot s{},p{};
    for(int i=0;i<31;i++) {
      d->qpos[m->jnt_qposadr[map.hinge[i]]]=.002*(i-15);
      d->qvel[m->jnt_dofadr[map.hinge[i]]]=.03*(i-15);
      if(map.backlash[i]>=0){d->qpos[m->jnt_qposadr[map.backlash[i]]]=.0001*(i+1);d->qvel[m->jnt_dofadr[map.backlash[i]]]=-.01*(i+1);}
    }
    map.extract(m,d,s);
    for(int i=0;i<31;i++) {
      require(std::abs(s.v[4][i]-(.002*(i-15)+(i<29?.0001*(i+1):0)))<1e-12,"output position map failed");
      require(std::abs(s.v[5][i]-(.03*(i-15)-(i<29?.01*(i+1):0)))<1e-12,"output velocity map failed");
      d->qpos[m->jnt_qposadr[map.hinge[i]]]+=.001;
      map.extract(m,d,p);
      for(int j=0;j<31;j++)require(std::abs(p.v[4][j]-s.v[4][j]-(i==j?.001:0))<1e-12,"independent SDK joint perturbation failed");
      d->qpos[m->jnt_qposadr[map.hinge[i]]] -= .001;
    }
    std::printf("{\"joints\":31,\"coaxial_pairs\":29,\"perturbation_checks\":31,\"physics_steps\":0,\"mujoco_version\":\"%s\"}\n",mj_versionString());
    mj_deleteData(d);mj_deleteModel(m);return 0;
  } catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
