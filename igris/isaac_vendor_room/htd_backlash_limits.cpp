// Native3.2.7 exact serialized-model audit for the explicit HTD15 limit variant.
// Reads models only: no stepping, ROS, controller or publication.
#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static void require(bool ok,const char* message) {
  if(!ok){std::cerr<<message<<"\n";std::exit(1);}
}
int main(int argc,char** argv) {
  require(argc==4 || argc==5,"Expected original/candidate/timeconstant[/explicit_upper_timeconstant]");
  require(!std::strcmp(mj_versionString(),"3.2.7"),"Requires supplied native MuJoCo3.2.7");
  char* end=nullptr;double tc=std::strtod(argv[3],&end);
  require(end && !*end && std::isfinite(tc) && tc==.005,"Only timeconstant0.005 allowed");
  const bool upper=argc==5;
  if(upper){char* end_upper=nullptr;const double u=std::strtod(argv[4],&end_upper);
    require(end_upper && !*end_upper && std::isfinite(u) && u==.005,"Only explicit upper timeconstant0.005 allowed");}
  char error[4096]{};
  auto a=mj_loadXML(argv[1],nullptr,error,sizeof(error));require(a,error);
  auto b=mj_loadXML(argv[2],nullptr,error,sizeof(error));require(b,error);
  require(a->njnt==b->njnt,"Joint count changed");
  std::vector<std::string> owned{"waist_yaw_backlash","waist_roll_backlash","waist_pitch_backlash"};
  for(const char* side:{"l","r"})for(const char* part:{"hip_pitch","hip_roll","hip_yaw","knee_pitch","ankle_pitch","ankle_roll"})
    owned.push_back(std::string(side)+"_"+part+"_backlash");
  if(upper)for(const char* side:{"l","r"})for(const char* part:{"shoulder_pitch","shoulder_roll","shoulder_yaw","elbow_pitch","wrist_yaw","wrist_roll","wrist_pitch"})
    owned.push_back(std::string(side)+"_"+part+"_backlash");
  std::vector<bool> selected(a->njnt,false);
  for(const auto& name:owned){
    int j=mj_name2id(a,mjOBJ_JOINT,name.c_str()),k=mj_name2id(b,mjOBJ_JOINT,name.c_str());
    require(j>=0 && j==k,"Owned joint missing or mapping changed");selected[j]=true;
    require(a->jnt_limited[j] && b->jnt_limited[j],"Owned backlash must retain limits");
    if(upper && (name.find("shoulder")!=std::string::npos || name.find("elbow")!=std::string::npos || name.find("wrist")!=std::string::npos)){
      require(a->jnt_type[j]==mjJNT_HINGE && b->jnt_type[j]==mjJNT_HINGE,"Upper backlash must remain a hinge");
      const double bound=name.find("wrist")!=std::string::npos?.01:.005;
      require(a->jnt_range[2*j]==-bound && a->jnt_range[2*j+1]==bound,"Unknown original upper backlash range");
    }
    require(a->jnt_solref[2*j]==.02 && a->jnt_solref[2*j+1]==1.,"Unknown original limit profile");
    require(b->jnt_solref[2*j]==tc && b->jnt_solref[2*j+1]==1.,"Unexpected owned limit profile");
  }
  for(int j=0;j<a->njnt;j++){
    if(selected[j])b->jnt_solref[2*j]=a->jnt_solref[2*j];
    else require(!std::memcmp(a->jnt_solref+2*j,b->jnt_solref+2*j,2*sizeof(mjtNum)),"Nonowned limit changed");
  }
  int size=mj_sizeModel(a);require(size==mj_sizeModel(b),"Serialized model size changed");
  std::vector<char> before(size),after(size);
  mj_saveModel(a,nullptr,before.data(),size);mj_saveModel(b,nullptr,after.data(),size);
  require(before==after,"Other serialized model parameters changed");
  std::cout<<"{\"mujoco_version\":\"3.2.7\",\"changed_joints\":"<<owned.size()<<",\"all_other_serialized_model_fields_equal\":true,\"changed_joint_names\":[";
  for(size_t i=0;i<owned.size();i++)std::cout<<(i?",":"")<<"\""<<owned[i]<<"\"";
  std::cout<<"]}\n";mj_deleteModel(a);mj_deleteModel(b);
}
