// Set room initial conditions after robot preparation, once per reset.
// No robot state, physics parameter, force, or state during replay is modified.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
[[noreturn]] void selector_error(const std::string& reason) {
 std::fprintf(stderr,"room staging selector: %s\n",reason.c_str());std::abort();
}
struct Selector {
 bool initialized=false,enabled=false;
 std::string scope="source_only";
 std::vector<std::string> names;
 std::vector<bool> selected;
 void read_once() {
  if(initialized)return;
  initialized=true;
  const char* configured_scope=std::getenv("IGRIS_ROOM_STAGE_SCOPE");
  if(configured_scope) {
   scope=configured_scope;
   if(scope!="source_only" && scope!="source_and_destination")selector_error("unknown scope: "+scope);
  }
  const char* path=std::getenv("IGRIS_ROOM_STAGE_BODIES_FILE");
  if(!path) {
   if(configured_scope)selector_error("explicit scope requires a body list");
   return;
  }
  enabled=true;std::ifstream file(path);
  if(!file)selector_error("cannot read body list");
  std::string line;
  while(std::getline(file,line)) {
   const auto first=line.find_first_not_of(" \t\r");
   if(first==std::string::npos)continue;
   const auto last=line.find_last_not_of(" \t\r");
   names.push_back(line.substr(first,last-first+1));
  }
  if(file.bad() || names.empty())selector_error("empty or unreadable body list");
 }
 void bind(const mjModel* m) {
  selected.assign(m->nbody,false);
  for(const auto& name:names) {
   int b=mj_name2id(m,mjOBJ_BODY,name.c_str());
   if(name.compare(0,5,"room_") || b<1 || name!=mj_id2name(m,mjOBJ_BODY,b))
    selector_error("unknown or non-room body: "+name);
   if(m->body_jntnum[b]!=1 || m->jnt_type[m->body_jntadr[b]]!=mjJNT_FREE)
    selector_error("body is not free: "+name);
   if(selected[b])selector_error("duplicate body: "+name);
   selected[b]=true;
  }
 }
};
void json_string(FILE* f,const std::string& value) {
 std::fputc('"',f);
 for(unsigned char c:value) {
  if(c=='"' || c=='\\') {std::fputc('\\',f);std::fputc(c,f);}
  else if(c<0x20)std::fprintf(f,"\\u%04x",c);
  else std::fputc(c,f);
 }
 std::fputc('"',f);
}
}

extern "C" void mj_step(const mjModel* m,mjData* d) {
 static auto step=reinterpret_cast<void(*)(const mjModel*,mjData*)>(dlsym(RTLD_NEXT,"mj_step"));
 static double last=-1.;static bool placed=false;
 static const mjModel* active_model=nullptr;static const mjData* active_data=nullptr;
 static Selector selector;
 const char* dir=std::getenv("IGRIS_ROOM_STAGE");
 if(dir) {
  selector.read_once();
  const bool new_model=active_model!=m;
  if(new_model)selector.bind(m);
  // mj_loadXML can call mj_step on compiler scratch data. Never reuse that
  // data's staging state for the runtime data, even when both start at t=0.
  bool reset=new_model || active_data!=d || last<0 || d->time<last;
  active_model=m;active_data=d;
  bool ready=access((std::string(dir)+"/room_layout.request").c_str(),F_OK)==0;
  if(reset)placed=false;
  if(reset || (ready && !placed)) {
   double tx=0,ty=0,yaw=0;
   if(ready) {
    FILE* f=std::fopen((std::string(dir)+"/room_layout.request").c_str(),"r");
    if(!f || std::fscanf(f,"%lf %lf %lf",&tx,&ty,&yaw)!=3 || !std::isfinite(tx+ty+yaw))std::abort();std::fclose(f);
   }
   int room_count=0;
   for(int b=1;b<m->nbody;b++) {
    const char* name=mj_id2name(m,mjOBJ_BODY,b);
    if(!name || std::strncmp(name,"room_",5) || m->body_jntnum[b]!=1)continue;
    int j=m->body_jntadr[b];if(m->jnt_type[j]!=mjJNT_FREE)std::abort();
    ++room_count;
    int q=m->jnt_qposadr[j],v=m->jnt_dofadr[j];
    std::memcpy(d->qpos+q,m->qpos0+q,7*sizeof(mjtNum));
    if(!ready)d->qpos[q]+=10.;
    else if(!selector.enabled || selector.selected[b]) {
     double x=d->qpos[q],y=d->qpos[q+1];
     d->qpos[q]=std::cos(yaw)*x-std::sin(yaw)*y+tx;d->qpos[q+1]=std::sin(yaw)*x+std::cos(yaw)*y+ty;
     mjtNum turn[4]={std::cos(yaw/2),0,0,std::sin(yaw/2)},quat[4];mju_mulQuat(quat,turn,d->qpos+q+3);mju_copy4(d->qpos+q+3,quat);
    }
    std::memset(d->qvel+v,0,6*sizeof(mjtNum));
    std::memset(d->qacc_warmstart+v,0,6*sizeof(mjtNum));
   }
   mj_forward(m,d);placed=ready;
   if(ready) {
    FILE* f=std::fopen((std::string(dir)+"/room_layout_applied.json").c_str(),"w");if(!f)std::abort();
    if(!selector.enabled) {
     // Retain the legacy task1 evidence format when no selector is supplied.
     std::fprintf(f,"{\"sim\":%.9f,\"translation_xy\":[%.9f,%.9f],\"yaw_rad\":%.9f,\"room_free_bodies\":39,\"robot_state_modified\":false,\"during_replay\":false}\n",d->time,tx,ty,yaw);
    } else {
     std::fprintf(f,"{\"sim\":%.9f,\"translation_xy\":[%.9f,%.9f],\"yaw_rad\":%.9f,\"room_free_bodies\":%d,\"robot_state_modified\":false,\"during_replay\":false,\"scope\":\"%s\",\"transforms_destination\":%s,\"transformed_body_count\":%zu,\"fixed_room_free_bodies\":%zu,\"transformed_bodies\":[",d->time,tx,ty,yaw,room_count,selector.scope.c_str(),selector.scope=="source_and_destination"?"true":"false",selector.names.size(),static_cast<size_t>(room_count)-selector.names.size());
     for(size_t i=0;i<selector.names.size();++i) {
      if(i)std::fputc(',',f);json_string(f,selector.names[i]);
     }
     std::fputs("]}\n",f);
    }
    std::fclose(f);
   }
  }
 }
 step(m,d);
 // Remember completed time so a reset after even one step is detected.
 if(dir)last=d->time;
}
