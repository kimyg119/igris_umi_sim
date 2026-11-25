// Run with the observer in LD_PRELOAD; see README for the native build command.
#include <mujoco/mujoco.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
int main() {
 const char* xml=R"(<mujoco><option timestep="0.001"/><worldbody>
 <geom name="eval_table_-1" type="plane" size="1 1 .1"/>
 <body name="eval_box" pos="0 0 .075" euler="0 0 20"><freejoint/>
 <geom name="eval_box_geom" type="box" size=".075 .075 .075" mass="1"/></body>
 <body name="l_hand_rcv" pos=".10 0 .10"><body>
 <joint name="l_hand_mcp_compliant" axis="0 1 0" stiffness="5" damping=".1"/>
 <geom type="sphere" size=".03" pos="0 0 -.03" mass=".1"/>
 </body></body></worldbody></mujoco>)";
 mjVFS vfs;mj_defaultVFS(&vfs);mj_addBufferVFS(&vfs,"test.xml",xml,std::string(xml).size());
 char error[1000];auto m=mj_loadXML("test.xml",&vfs,error,sizeof(error));
 if(!m){std::fprintf(stderr,"%s\n",error);return 1;}
 auto d=mj_makeData(m);
 std::fclose(std::fopen(std::getenv("IGRIS_EVAL_TRIGGER"),"w"));
 for(int i=0;i<1000;i++)mj_step(m,d);
 usleep(40000);mj_step(m,d);
 auto f=std::fopen((std::string(std::getenv("IGRIS_EVAL_LOG"))+".wrenches").c_str(),"rb");assert(f);
 double w[21],last=-1,force[3]={},moment[3]={},mcp=0;
 while(std::fread(w,sizeof(double),21,f)==21) {
  if(w[0]!=last){last=w[0];mju_zero3(force);mju_zero3(moment);mcp=0;}
  mjtNum arm[3],torque[3];mju_sub3(arm,w+4,w+7);mju_cross(torque,arm,w+10);mju_addTo3(torque,w+13);
  mju_addTo3(force,w+10);mju_addTo3(moment,torque);mcp+=w[16];
 }
 int body=mj_name2id(m,mjOBJ_BODY,"eval_box");mjtNum local[3];mju_mulMatTVec(local,d->xmat+9*body,moment,3,3);
 for(int k=0;k<3;k++){assert(std::abs(force[k]-d->qfrc_constraint[k])<1e-8);assert(std::abs(local[k]-d->qfrc_constraint[3+k])<1e-8);}
 assert(std::abs(mcp-d->qfrc_constraint[6])<1e-8);
 assert(std::abs(mcp)>.001); // Exercise an actual moment arm, not just the unloaded case.
 std::printf("PASS: native world wrench and MCP projection match constraint forces; Fz=%.6f N, MCP=%.6f Nm\n",force[2],mcp);
 std::fclose(f);mj_deleteData(d);mj_deleteModel(m);mj_deleteVFS(&vfs);
}
