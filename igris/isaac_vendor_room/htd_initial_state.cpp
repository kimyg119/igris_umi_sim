// Explicit, one-shot empty-hand initial data at an observed native reset.
// Never changes a model, room body, actuator, or subsequent policy state.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
[[noreturn]] void fail(const std::string& s){std::fprintf(stderr,"HTD_INITIAL_STATE: %s\n",s.c_str());std::abort();}
void require(bool x,const char*s){if(!x)fail(s);}
int qsize(int type){return type==mjJNT_FREE?7:type==mjJNT_BALL?4:1;}
int vsize(int type){return type==mjJNT_FREE?6:type==mjJNT_BALL?3:1;}
bool room(const mjModel*m,int joint){const char*n=mj_id2name(m,mjOBJ_BODY,m->jnt_bodyid[joint]);return n&&!std::strncmp(n,"room_",5);}
struct State {
 std::vector<int> joints;std::vector<std::vector<double>> values;
 State(const mjModel*m,const std::string&path){
  std::ifstream f(path);int count=0;f>>count;require(bool(f)&&count>0&&count<=m->njnt,"invalid initial joint count");
  std::vector<bool> seen(m->njnt,false);
  for(int i=0;i<count;i++){
   std::string name;int size;f>>name>>size;int j=mj_name2id(m,mjOBJ_JOINT,name.c_str());
   require(bool(f)&&j>=0&&!seen[j],"missing or duplicate initial joint");require(!room(m,j),"room state cannot be initialized");
   require(size==qsize(m->jnt_type[j]),"initial joint coordinate size mismatch");
   std::vector<double> q(size);for(double&v:q){f>>v;require(bool(f)&&std::isfinite(v),"nonfinite/incomplete initial coordinates");}
   if(size==4||size==7){int offset=size==7?3:0;double norm=0;for(int k=offset;k<offset+4;k++)norm+=q[k]*q[k];require(std::abs(norm-1)<1e-8,"initial quaternion is not unit length");}
   seen[j]=true;joints.push_back(j);values.push_back(q);
  }
  std::string trailing;require(!(f>>trailing),"trailing initial state records");
  for(int j=0;j<m->njnt;j++)require(seen[j]!=room(m,j),"initial state must cover every robot joint and no room joints");
 }
 void apply(const mjModel*m,mjData*d)const{
  for(size_t i=0;i<joints.size();i++){
   int j=joints[i],q=m->jnt_qposadr[j],v=m->jnt_dofadr[j];
   for(size_t k=0;k<values[i].size();k++)d->qpos[q+k]=values[i][k];
   for(int k=0;k<vsize(m->jnt_type[j]);k++){d->qvel[v+k]=0;d->qacc_warmstart[v+k]=0;}
  }
  mj_forward(m,d);
 }
};
}

extern "C" void mj_step(const mjModel*m,mjData*d){
 static auto next=reinterpret_cast<void(*)(const mjModel*,mjData*)>(dlsym(RTLD_NEXT,"mj_step"));
 require(next,"native mj_step missing");const char*configured=std::getenv("IGRIS_HTD_INITIAL_STATE_DIR");
 if(!configured){next(m,d);return;}
 static const mjModel*bound_model=nullptr;static mjData*bound_data=nullptr;static double last=-1;static bool applied=false;
 std::string dir(configured);bool ready=access((dir+"/htd_initial_state.request").c_str(),F_OK)==0;
 bool same=bound_model==m&&bound_data==d;
 bool reset=same&&d->time<last-1e-9;
 if(ready&&!same)fail("initialization request predates runtime or data ownership changed");
 if(applied&&reset)fail("second reset after one-shot initialization is forbidden");
 if(ready&&!applied&&reset){
  require(last>=1.,"initialization requires observed pre-reset runtime");
  std::ifstream request(dir+"/htd_initial_state.request");std::string token,extra;request>>token;
  require(token=="initialize-on-empty-reset"&&!(request>>extra),"invalid initialization request");
  require(access((dir+"/htd_initial_state_applied.json").c_str(),F_OK)!=0,"initialization acknowledgement already exists");
  State state(m,dir+"/htd_initial_state.txt");
  std::vector<double>before(d->qpos,d->qpos+m->nq);double stamp=d->time;state.apply(m,d);
  for(int j=0;j<m->njnt;j++)if(room(m,j))for(int k=0;k<qsize(m->jnt_type[j]);k++)
    require(d->qpos[m->jnt_qposadr[j]+k]==before[m->jnt_qposadr[j]+k],"room state changed");
  require(stamp==d->time,"native clock changed");applied=true;
  std::string temporary=dir+"/htd_initial_state_applied.tmp";FILE*f=std::fopen(temporary.c_str(),"w");require(f,"cannot record initialization");
  std::fprintf(f,"{\"schema\":\"native_empty_initial_state/v1\",\"sim\":%.17g,\"prior_sim\":%.17g,\"robot_joints\":%zu,\"apply_count\":1,\"robot_velocity_zero\":true,\"room_qpos_unchanged\":true,\"model_parameters_unchanged\":true}\n",d->time,last,state.joints.size());
  require(std::fclose(f)==0&&std::rename(temporary.c_str(),(dir+"/htd_initial_state_applied.json").c_str())==0,"cannot publish initialization acknowledgement");
 }
 bound_model=m;bound_data=d;next(m,d);last=d->time;
}

#ifdef IGRIS_INITIAL_STATE_AUDIT
int main(int argc,char**argv){
 require(argc==3,"expected model and initial state");char error[4096];auto m=mj_loadXML(argv[1],nullptr,error,sizeof(error));require(m,error);
 auto d=mj_makeData(m);State s(m,argv[2]);s.apply(m,d);
 double residual=0;for(int k=0;k<d->nefc;k++)if(d->efc_type[k]==mjCNSTR_EQUALITY)residual=std::fmax(residual,std::abs(d->efc_pos[k]));
 std::printf("{\"mujoco\":\"%s\",\"robot_joints\":%zu,\"maximum_equality_residual\":%.17g,\"physics_steps\":0}\n",mj_versionString(),s.joints.size(),residual);
 mj_deleteData(d);mj_deleteModel(m);return 0;
}
#endif
