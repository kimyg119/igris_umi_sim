// Read-only observer of the vendor MuJoCo physics. No state or control writes.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>
static double wall() { timespec t; clock_gettime(CLOCK_REALTIME,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static bool descendant(const mjModel* m,int body,int ancestor) {
 while(body>0) { if(body==ancestor)return true;body=m->body_parentid[body]; }
 return false;
}
extern "C" void mj_step(const mjModel* m,mjData* d) {
 static auto step=(void(*)(const mjModel*,mjData*))dlsym(RTLD_NEXT,"mj_step");
 step(m,d);
 const char* trigger=getenv("IGRIS_EVAL_TRIGGER");
 if(!trigger || access(trigger,F_OK)) return;
 static FILE* f=nullptr; static FILE* patches=nullptr; static FILE* pads=nullptr; static FILE* offsets=nullptr; static double last=0;
 double now=wall(); if(now-last<1./30.)return; last=now;
 if(!f) { const char* path=getenv("IGRIS_EVAL_LOG"); if(!path)return; f=fopen(path,"wb"); if(!f)return; int h[]={m->nq,m->nv}; fwrite(h,sizeof(int),2,f); patches=fopen((std::string(path)+".contacts").c_str(),"wb"); pads=fopen((std::string(path)+".pads").c_str(),"wb"); offsets=fopen((std::string(path)+".offsets").c_str(),"wb"); }
 static int box=mj_name2id(m,mjOBJ_GEOM,"eval_box_geom");
 static int left=mj_name2id(m,mjOBJ_BODY,"l_hand_rcv"),right=mj_name2id(m,mjOBJ_BODY,"r_hand_rcv");
 static int ta=mj_name2id(m,mjOBJ_GEOM,"eval_table_-1"),tb=mj_name2id(m,mjOBJ_GEOM,"eval_table_1");
 static const char* offset_names[31]={"waist_yaw_backlash","waist_roll_backlash","waist_pitch_backlash","l_hip_pitch_backlash","l_hip_roll_backlash","l_hip_yaw_backlash","l_knee_pitch_backlash","l_ankle_pitch_backlash","l_ankle_roll_backlash","r_hip_pitch_backlash","r_hip_roll_backlash","r_hip_yaw_backlash","r_knee_pitch_backlash","r_ankle_pitch_backlash","r_ankle_roll_backlash","l_shoulder_pitch_backlash","l_shoulder_roll_backlash","l_shoulder_yaw_backlash","l_elbow_pitch_backlash","l_wrist_yaw_backlash","l_wrist_roll_backlash","l_wrist_pitch_backlash","r_shoulder_pitch_backlash","r_shoulder_roll_backlash","r_shoulder_yaw_backlash","r_elbow_pitch_backlash","r_wrist_yaw_backlash","r_wrist_roll_backlash","r_wrist_pitch_backlash","neck_yaw_backlash","neck_pitch_backlash"};
 double offset_row[32]={now};
 for(int j=0;j<31;j++){int id=mj_name2id(m,mjOBJ_JOINT,offset_names[j]);if(id>=0)offset_row[j+1]=d->qpos[m->jnt_qposadr[id]];}
 if(offsets){fwrite(offset_row,sizeof(double),32,offsets);fflush(offsets);}
 // Gain diagnostics: native actuator controls/forces, distinct from SDK joint errors.
 static FILE* effort=nullptr;
 if(!effort){effort=fopen((std::string(getenv("IGRIS_EVAL_LOG"))+".effort").c_str(),"wb");if(effort){int h[]={m->nu,m->nv};fwrite(h,sizeof(int),2,effort);}}
 if(effort){std::vector<double> e={now,d->time};e.insert(e.end(),d->ctrl,d->ctrl+m->nu);e.insert(e.end(),d->actuator_force,d->actuator_force+m->nu);e.insert(e.end(),d->qfrc_actuator,d->qfrc_actuator+m->nv);fwrite(e.data(),sizeof(double),e.size(),effort);fflush(effort);}
 double lf=0,rf=0,other=0,self=0,depth=0,table_robot=0,box_environment=0;
 double patch_forces[7]={now,0,0,0,0,0,0}; // time, left palm/tip/other, right palm/tip/other
 double pad_forces[3]={now,0,0}; // time, left/right MCP proximal plate (not distal fingertip)
 static FILE* wrenches=nullptr;
 if(!wrenches)wrenches=fopen((std::string(getenv("IGRIS_EVAL_LOG"))+".wrenches").c_str(),"wb");
 for(int i=0;i<d->ncon;i++) {
  const auto& c=d->contact[i]; int a=c.geom[0],b=c.geom[1]; if(a<0||b<0)continue;
  mjtNum force[6]; mj_contactForce(m,d,i,force);
  if(a==box || b==box) { int body=m->geom_bodyid[a==box?b:a]; if(descendant(m,body,left))lf+=force[0];else if(descendant(m,body,right))rf+=force[0];else { other+=force[0]; if(m->geom_bodyid[a==box?b:a]==0)box_environment+=force[0]; } }
  if(a==box || b==box) {
   int geom=a==box?b:a,body=m->geom_bodyid[geom];
   bool is_left=descendant(m,body,left),is_right=descendant(m,body,right);
   const char* mesh=m->geom_type[geom]==mjGEOM_MESH ? mj_id2name(m,mjOBJ_MESH,m->geom_dataid[geom]):nullptr;
   int surface=mesh && strstr(mesh,"___1_5") ? 0 : mesh && strstr(mesh,"___1_4") ? 1 : mesh && strstr(mesh,"___1_6") ? 2 : 3;
   if(wrenches) {
    // Contact-frame rows are world axes; force acts on geom[1].
    // Record the wrench ON the box, plus the opposite generalized hand loads.
    mjtNum fworld[3],tworld[3];mju_mulMatTVec(fworld,c.frame,force,3,3);mju_mulMatTVec(tworld,c.frame,force+3,3,3);
    double sign=b==box?1.:-1.;mju_scl3(fworld,fworld,sign);mju_scl3(tworld,tworld,sign);
    const mjtNum* com=d->xipos+3*m->geom_bodyid[box];
    int group=is_left?surface:is_right?4+surface:m->geom_bodyid[geom]==0?8:9;
    double joint_load[2]={0,0};
    if(is_left || is_right)for(int k=0;k<2;k++) {
     std::string name=std::string(is_left?"l":"r")+(k==0?"_hand_mcp_compliant":"_hand_dip_spring");
     int j=mj_name2id(m,mjOBJ_JOINT,name.c_str());
     if(j>=0 && descendant(m,body,m->jnt_bodyid[j])) {
      mjtNum arm[3],moment[3];mju_sub3(arm,c.pos,d->xanchor+3*j);mju_cross(moment,arm,fworld);mju_addTo3(moment,tworld);
      joint_load[k]=-mju_dot3(moment,d->xaxis+3*j);
     }
    }
    double w[]={now,d->time,(double)group,force[0],c.pos[0],c.pos[1],c.pos[2],com[0],com[1],com[2],
                fworld[0],fworld[1],fworld[2],tworld[0],tworld[1],tworld[2],joint_load[0],joint_load[1],
                sign*c.frame[0],sign*c.frame[1],sign*c.frame[2]};
    fwrite(w,sizeof(double),21,wrenches);
   }
   if(is_left || is_right) {
    int part=mesh && strstr(mesh,"___1_5") ? 0 : mesh && strstr(mesh,"___1_4") ? 1 : 2;
    patch_forces[1+(is_left?0:3)+part]+=force[0];
    if(mesh && strstr(mesh,"___1_6"))pad_forces[is_left?1:2]+=force[0];
   }
  }
  else if((a==ta || a==tb || b==ta || b==tb) && m->geom_bodyid[a==ta || a==tb ? b:a]>0) table_robot+=force[0];
  else if(m->geom_bodyid[a]>0 && m->geom_bodyid[b]>0 && force[0]>.1) { self++;depth=std::min(depth,(double)c.dist); }
 }
 if(wrenches)fflush(wrenches);
 std::vector<double> row={now,d->time};row.insert(row.end(),d->qpos,d->qpos+m->nq);row.insert(row.end(),d->qvel,d->qvel+m->nv);
 row.insert(row.end(),{lf,rf,other,self,depth,table_robot,box_environment});fwrite(row.data(),sizeof(double),row.size(),f);fflush(f);
 if(patches) { fwrite(patch_forces,sizeof(double),7,patches);fflush(patches); }
 if(pads) { fwrite(pad_forces,sizeof(double),3,pads);fflush(pads); }
}
