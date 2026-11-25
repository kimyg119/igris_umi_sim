// Native3.2.7 model-only audit: exactly two passive DIP damping coefficients.
// No mjData allocation, stepping, ROS or controller calls.
#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

static void require(bool ok,const char* message) {
  if(!ok){std::cerr<<message<<"\n";std::exit(1);}
}
int main(int argc,char** argv) {
  require(argc==3,"Expected original/candidate model paths");
  require(!std::strcmp(mj_versionString(),"3.2.7"),"Requires native MuJoCo3.2.7");
  char error[4096]{};
  auto a=mj_loadXML(argv[1],nullptr,error,sizeof(error));require(a,error);
  auto b=mj_loadXML(argv[2],nullptr,error,sizeof(error));require(b,error);
  require(a->njnt==b->njnt && a->nv==b->nv,"Joint/dof counts changed");
  for(const char* name:{"l_hand_dip_spring","r_hand_dip_spring"}) {
    int j=mj_name2id(a,mjOBJ_JOINT,name),k=mj_name2id(b,mjOBJ_JOINT,name);
    require(j>=0 && j==k,"DIP mapping changed");
    require(a->jnt_type[j]==mjJNT_HINGE && b->jnt_type[k]==mjJNT_HINGE,"DIP must stay a hinge");
    require(!a->jnt_limited[j] && !b->jnt_limited[k],"DIP must retain no hard stop");
    int v=a->jnt_dofadr[j];require(v==b->jnt_dofadr[k],"DIP dof mapping changed");
    require(a->dof_damping[v]==.0001,"Unknown original DIP damping");
    require(b->dof_damping[v]==.002,"Unexpected requested DIP damping");
    b->dof_damping[v]=a->dof_damping[v];
  }
  int size=mj_sizeModel(a);require(size==mj_sizeModel(b),"Model size changed");
  std::vector<char> before(size),after(size);
  mj_saveModel(a,nullptr,before.data(),size);mj_saveModel(b,nullptr,after.data(),size);
  require(before==after,"Other serialized model fields changed");
  std::cout<<"{\"mujoco_version\":\"3.2.7\",\"physics_steps\":0,\"changed_dofs\":2,"
    "\"changed_joint_names\":[\"l_hand_dip_spring\",\"r_hand_dip_spring\"],"
    "\"original_damping_Nms_rad\":0.0001,\"requested_damping_Nms_rad\":0.002,"
    "\"all_other_serialized_model_fields_equal\":true}\n";
  mj_deleteModel(a);mj_deleteModel(b);
}
