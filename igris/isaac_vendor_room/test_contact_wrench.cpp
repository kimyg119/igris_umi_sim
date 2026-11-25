// Native regression: body/geom order and contact frame must not invert world load.
#include "contact_wrench.h"
#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>
int main(){
 const char* xml="<mujoco><option timestep='.001' gravity='-5.886 0 -7.848'/><worldbody><geom name='ground' type='plane' size='1 1 .1' quat='.9486832980505138 0 .31622776601683794 0'/><body name='payload' pos='.05994 0 .07992'><freejoint/><geom name='cube' type='box' size='.1 .1 .1' mass='1' quat='.9486832980505138 0 .31622776601683794 0'/></body></worldbody></mujoco>";
 mjVFS vfs;mj_defaultVFS(&vfs);mj_addBufferVFS(&vfs,"test.xml",xml,int(strlen(xml)));char error[1024];
 auto*m=mj_loadXML("test.xml",&vfs,error,sizeof(error));assert(m);auto*d=mj_makeData(m);
 for(int i=0;i<3000;i++)mj_step(m,d);
 int payload=mj_name2id(m,mjOBJ_BODY,"payload");std::vector<mjtNum> before(mj_stateSize(m,mjSTATE_INTEGRATION)),after(before.size());mj_getState(m,d,before.data(),mjSTATE_INTEGRATION);
 double force[3]{},reverse[3]{};int count=0;
 for(int c=0;c<d->ncon;c++){
  auto w=payload_contact_wrench(m,d,c,payload);auto r=payload_contact_wrench(m,d,c,0);
  for(int k=0;k<3;k++){force[k]+=w.force_world[k];reverse[k]+=r.force_world[k];assert(fabs(w.relative_velocity_world[k])<1e-8);}
  count++;
 }
 mj_getState(m,d,after.data(),mjSTATE_INTEGRATION);assert(before==after);assert(count>0);
 const double expected[]={5.886,0,7.848};for(int k=0;k<3;k++){assert(fabs(force[k]-expected[k])<1e-6);assert(fabs(reverse[k]+expected[k])<1e-6);}
 printf("PASS: inclined 1 kg load, both body orders, read-only state (%d contacts)\n",count);
 mj_deleteData(d);mj_deleteModel(m);mj_deleteVFS(&vfs);
}
