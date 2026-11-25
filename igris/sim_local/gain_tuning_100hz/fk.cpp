// Offline FK only, using the same vendor MuJoCo library as the integrated runs.
#include <mujoco/mujoco.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc,char**argv) {
  if(argc!=2)return 2;
  std::string dir=argv[1];char error[2048];
  mjModel* m=mj_loadXML((dir+"/scene.xml").c_str(),nullptr,error,sizeof(error));
  if(!m){fprintf(stderr,"%s\n",error);return 3;}
  mjData* d=mj_makeData(m);
  FILE* p=fopen((dir+"/physics.bin").c_str(),"rb");
  FILE* c=fopen((dir+"/commands.bin").c_str(),"rb");
  if(!p||!c)return 4;
  int h[3];if(fread(h,sizeof(int),3,p)!=3)return 4;
  if(h[0]!=m->nq||h[1]!=m->nv||h[2]!=m->nu)return 5;
  std::vector<double> cmds;double crow[18];
  while(fread(crow,sizeof(double),18,c)==18)cmds.insert(cmds.end(),crow,crow+18);
  fclose(c);size_t nc=cmds.size()/18;if(!nc)return 6;
  const char* names[]={"waist_yaw","waist_roll","waist_pitch","l_shoulder_pitch","l_shoulder_roll","l_shoulder_yaw","l_elbow_pitch","l_wrist_yaw","l_wrist_roll","l_wrist_pitch","r_shoulder_pitch","r_shoulder_roll","r_shoulder_yaw","r_elbow_pitch","r_wrist_yaw","r_wrist_roll","r_wrist_pitch"};
  int addresses[17],offsets[17],body[2];
  for(int i=0;i<17;++i){
    int j=mj_name2id(m,mjOBJ_JOINT,names[i]);
    int k=mj_name2id(m,mjOBJ_JOINT,(std::string(names[i])+"_backlash").c_str());
    if(j<0||k<0)return 7;addresses[i]=m->jnt_qposadr[j];offsets[i]=m->jnt_qposadr[k];
  }
  body[0]=mj_name2id(m,mjOBJ_BODY,"end_effector_left");body[1]=mj_name2id(m,mjOBJ_BODY,"end_effector_right");
  if(body[0]<0||body[1]<0)return 8;
  FILE* o=fopen((dir+"/ee.bin").c_str(),"wb");
  int width=2+h[0]+h[1]+2*h[2]+16;std::vector<double> row(width);size_t ci=0;
  while(fread(row.data(),sizeof(double),width,p)==(size_t)width){
    if(row[0]<cmds[0]||row[0]>cmds[(nc-1)*18])continue;
    while(ci+1<nc&&cmds[(ci+1)*18]<=row[0])++ci;
    memcpy(d->qpos,row.data()+2,m->nq*sizeof(double));mj_kinematics(m,d);
    std::vector<double> r={row[0],row[1]};
    auto append=[&](){for(int b:body){r.insert(r.end(),d->xpos+3*b,d->xpos+3*b+3);r.insert(r.end(),d->xquat+4*b,d->xquat+4*b+4);}};
    append();
    for(int j=0;j<17;++j){d->qpos[addresses[j]]=cmds[ci*18+1+j];d->qpos[offsets[j]]=0;}
    mj_kinematics(m,d);append();fwrite(r.data(),sizeof(double),r.size(),o);
  }
  fclose(o);fclose(p);mj_deleteData(d);mj_deleteModel(m);
}
