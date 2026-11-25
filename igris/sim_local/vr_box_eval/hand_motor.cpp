// Simulation-only MCP inner position servo. Body/leg control stays with the vendor.
// Loaded separately from the read-only physics observer; never writes qpos/qvel.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <string>

static double wall() { timespec t; clock_gettime(CLOCK_REALTIME,&t); return t.tv_sec+t.tv_nsec*1e-9; }
extern "C" void mj_step(const mjModel* m, mjData* d) {
 static auto step=(void(*)(const mjModel*,mjData*))dlsym(RTLD_NEXT,"mj_step");
 const char* directory=getenv("IGRIS_HAND_DIR");
 if(!directory) { step(m,d); return; }
 static int joints[2]={mj_name2id(m,mjOBJ_JOINT,"l_hand_mcp_compliant"),mj_name2id(m,mjOBJ_JOINT,"r_hand_mcp_compliant")};
 static int dips[2]={mj_name2id(m,mjOBJ_JOINT,"l_hand_dip_spring"),mj_name2id(m,mjOBJ_JOINT,"r_hand_dip_spring")};
 static int config=mj_name2id(m,mjOBJ_NUMERIC,"hand_motor_config");
 if(joints[0]<0 || joints[1]<0 || dips[0]<0 || dips[1]<0 || config<0) { std::fprintf(stderr,"Missing compliant hand model\n"); std::abort(); }
 const mjtNum* cfg=m->numeric_data+m->numeric_adr[config]; // rest, supply A, stall Nm/A, V, P raw, PWM raw, no-load rad/s
 static double reference[2]={cfg[0],cfg[0]}, read_at=-1, logged_at=-1, logged_wall=-1, command_stamp=0;
 static FILE* log=std::fopen((std::string(directory)+"/hand_motor.bin").c_str(),"wb");
 double now=wall();
 if(d->time<read_at || d->time-read_at>=.02) {
  read_at=d->time;
  FILE* command=std::fopen((std::string(directory)+"/hand_reference.txt").c_str(),"r");
  if(command) {
   double stamp,left,right;
   if(std::fscanf(command,"%lf %lf %lf",&stamp,&left,&right)==3 && std::isfinite(left) && std::isfinite(right)
      && now-stamp>=0 && now-stamp<.1 && left>=0 && right>=0 && left<=1.0471975512 && right<=1.0471975512) {
    reference[0]=left;reference[1]=right;command_stamp=stamp;
   }
   std::fclose(command);
  }
 }
 double torque[2];
 for(int i=0;i<2;i++) {
  int q=m->jnt_qposadr[joints[i]],v=m->jnt_dofadr[joints[i]];
  double voltage_torque=cfg[2]*(cfg[5]/128.)*(4096./(2*M_PI))*(reference[i]-d->qpos[q])/885.;
  double pwm_torque=cfg[2]*cfg[6]/885.;
  double requested=std::clamp(voltage_torque,-pwm_torque,pwm_torque)-cfg[2]/cfg[7]*d->qvel[v];
  double a=cfg[3]/(cfg[2]*cfg[2]),b=std::max((requested>=0 ? 1. : -1.)*d->qvel[v],0.)/cfg[4];
  double limit=(-b+std::sqrt(b*b+4*a*cfg[1]))/(2*a);
  torque[i]=std::clamp(requested,-limit,limit);
  d->qfrc_applied[v]=torque[i]; // Only these two extra joints; no vendor channels are modified.
 }
 step(m,d);
 double observed_wall=wall();
 if(log && (d->time<logged_at || d->time-logged_at>=.02 ||
            (getenv("IGRIS_HAND_LOG_WALL") && observed_wall-logged_wall>=.02))) {
  logged_at=d->time;logged_wall=observed_wall;
  double row[23]={observed_wall,d->time,command_stamp};
  for(int i=0;i<2;i++) {
   int q=m->jnt_qposadr[joints[i]],v=m->jnt_dofadr[joints[i]];
   double* item=row+3+6*i;
   item[0]=d->qpos[q];item[1]=d->qvel[v];item[2]=torque[i];
   item[3]=d->qfrc_bias[v]-d->qfrc_passive[v];item[4]=reference[i];item[5]=d->qfrc_constraint[v];
   q=m->jnt_qposadr[dips[i]];v=m->jnt_dofadr[dips[i]];
   row[15+4*i]=d->qpos[q];row[16+4*i]=d->qvel[v];
   row[17+4*i]=d->qfrc_passive[v];row[18+4*i]=d->qfrc_constraint[v];
  }
  std::fwrite(row,sizeof(double),23,log);std::fflush(log);
 }
}
