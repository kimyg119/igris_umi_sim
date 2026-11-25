// Static contact probes with the actual vendor library; no controller or stepping.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <string>
#include <vector>
int main(int argc,char** argv) {
  if((argc<4 || argc>9) || std::strcmp(mj_versionString(),"3.2.7"))return 2;
  const bool torsion_override=argc==9;double palm_torsion=0.;
  if(torsion_override) {
    char* end=nullptr;palm_torsion=std::strtod(argv[8],&end);
    if(end==argv[8] || *end || !std::isfinite(palm_torsion) || palm_torsion<=0 || palm_torsion>.05)return 2;
  }
  const bool feet=argc>=6 && !std::strcmp(argv[5],"with-feet");
  if(argc>=6 && !feet && std::strcmp(argv[5],"without-feet"))return 2;
  char error[4096]{}; auto m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!m){std::fprintf(stderr,"%s\n",error);return 1;}
  auto d=mj_makeData(m);const double mu=std::atof(argv[3]);
  const double tau=argc>=5?std::atof(argv[4]):0.;int refined_pairs=0,foot_pairs=0;double friction_reference_error=0;
  const double hand_tau=argc>=7?std::atof(argv[6]):tau;
  int expected_support_pairs=91;
  if(argc>=8) {
    char* end=nullptr;long count=std::strtol(argv[7],&end,10);
    if(!end || *end || count<1 || count>10000)return 2;
    expected_support_pairs=static_cast<int>(count);
  }
  int pairs=0,probes=0;double max_error=0;
  int palm_pairs=0,original_torsion_pairs=0;
  struct TorsionProbe{std::string hand;std::array<double,5> friction{};};
  std::vector<TorsionProbe> torsion_probes;
  for(int p=0;p<m->npair;p++) {
    auto name=mj_id2name(m,mjOBJ_PAIR,p);
    if(tau && name && (!std::strncmp(name,"paper_silicone_",15)||!std::strncmp(name,"box_friction_",13)||!std::strncmp(name,"foot_friction_",14))){
      const double expected=!std::strncmp(name,"paper_silicone_",15)?hand_tau:tau;
      refined_pairs++;friction_reference_error=std::max(friction_reference_error,std::abs(m->pair_solreffriction[2*p]-expected));
    }
    if(name && !std::strncmp(name,"foot_friction_",14)){
      foot_pairs++;int foot=m->pair_geom1[p],floor=m->pair_geom2[p];
      if(m->geom_priority[foot]<m->geom_priority[floor])std::swap(foot,floor);
      if(m->geom_priority[foot]<=m->geom_priority[floor] || m->pair_dim[p]!=m->geom_condim[foot])return 1;
      for(int k=0;k<2;k++)max_error=std::max(max_error,std::abs(m->pair_solref[2*p+k]-m->geom_solref[2*foot+k]));
      for(int k=0;k<5;k++)max_error=std::max(max_error,std::abs(m->pair_solimp[5*p+k]-m->geom_solimp[5*foot+k]));
      const int components[5]={0,0,1,2,2};
      for(int k=0;k<5;k++)max_error=std::max(max_error,std::abs(m->pair_friction[5*p+k]-m->geom_friction[3*foot+components[k]]));
    }
    if(!name || std::strncmp(name,"paper_silicone_",15))continue;
    pairs++;int h=m->pair_geom1[p],b=m->pair_geom2[p];
    auto n=mj_id2name(m,mjOBJ_GEOM,h);
    if(!n || (std::strncmp(n,"l_hand",6)&&std::strncmp(n,"r_hand",6)))std::swap(h,b);
    const char* hand_name=mj_id2name(m,mjOBJ_GEOM,h);
    const bool palm=hand_name && (!std::strcmp(hand_name,"l_hand_본체1_5_geom")||!std::strcmp(hand_name,"r_hand_본체1_5_geom"));
    const double expected_spin=palm&&torsion_override?palm_torsion:std::max(m->geom_friction[3*h+1],m->geom_friction[3*b+1]);
    const double expected_rolling=std::max(m->geom_friction[3*h+2],m->geom_friction[3*b+2]);
    if(torsion_override) {
      (palm?palm_pairs:original_torsion_pairs)++;
      if(m->pair_dim[p]!=4)return 1;
      max_error=std::max(max_error,std::abs(m->pair_friction[5*p+2]-expected_spin));
      for(int k=3;k<5;k++)max_error=std::max(max_error,std::abs(m->pair_friction[5*p+k]-expected_rolling));
    }
    const int body=m->geom_bodyid[b],joint=m->body_jntadr[body];
    if(m->jnt_type[joint]!=mjJNT_FREE)return 1;
    for(int k=0;k<2;k++)max_error=std::max(max_error,std::abs(m->pair_friction[5*p+k]-mu));
    mj_resetDataKeyframe(m,d,0);mj_forward(m,d);
    int q=m->jnt_qposadr[joint];
    for(int k=0;k<3;k++)d->qpos[q+k]=d->geom_xpos[3*h+k];
    mj_forward(m,d);bool found=false;TorsionProbe probe;probe.hand=hand_name?hand_name:"";
    for(int c=0;c<d->ncon;c++) {
      const auto& ct=d->contact[c];
      if(!((ct.geom1==h && ct.geom2==b)||(ct.geom1==b && ct.geom2==h)))continue;
      found=true;for(int k=0;k<2;k++)max_error=std::max(max_error,std::abs(ct.friction[k]-mu));
      if(torsion_override) {
        if(ct.dim!=4)return 1;
        max_error=std::max(max_error,std::abs(ct.friction[2]-expected_spin));
        for(int k=3;k<5;k++)max_error=std::max(max_error,std::abs(ct.friction[k]-expected_rolling));
        std::copy_n(ct.friction,5,probe.friction.begin());
      }
      if(tau)friction_reference_error=std::max(friction_reference_error,std::abs(ct.solreffriction[0]-hand_tau));
    }
    probes+=found;
    if(torsion_override&&found)torsion_probes.push_back(probe);
  }
  bool ok=pairs==76 && probes==pairs && max_error<1e-12 && foot_pairs==(feet?2:0) && (!torsion_override||(palm_pairs==2&&original_torsion_pairs==74)) && (!tau || (refined_pairs==76+expected_support_pairs+(feet?2:0) && friction_reference_error<1e-12));
  FILE* f=std::fopen(argv[2],"w");if(!f)return 1;
  std::fprintf(f,"{\"pass\":%s,\"mujoco_version\":\"%s\",\"pairs\":%d,\"contact_probes\":%d,\"mu\":%.17g,\"max_error\":%.17g,\"pair_friction_time_constant_s\":%.17g,\"hand_friction_time_constant_s\":%.17g,\"refined_pairs\":%d,\"friction_reference_max_error\":%.17g,\"physics_steps\":0",ok?"true":"false",mj_versionString(),pairs,probes,mu,max_error,tau,hand_tau,refined_pairs,friction_reference_error);
  if(torsion_override) {
    std::fprintf(f,",\"palm_torsion_m\":%.17g,\"palm_torsion_pairs\":%d,\"original_torsion_pairs\":%d,\"torsion_probes\":[",palm_torsion,palm_pairs,original_torsion_pairs);
    for(size_t i=0;i<torsion_probes.size();i++) {
      const auto& probe=torsion_probes[i];std::fprintf(f,"%s{\"hand_geom\":\"%s\",\"contact_friction\":[",i?",":"",probe.hand.c_str());
      for(int k=0;k<5;k++)std::fprintf(f,"%s%.17g",k?",":"",probe.friction[k]);std::fprintf(f,"]}");
    }
    std::fprintf(f,"]");
  }
  std::fprintf(f,"}\n");
  std::fclose(f);mj_deleteData(d);mj_deleteModel(m);return ok?0:1;
}
