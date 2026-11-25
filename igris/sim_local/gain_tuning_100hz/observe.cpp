// Simulation-only observer. Calls the vendor mj_step unchanged, then records.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

extern "C" void mj_step(const mjModel* m, mjData* d) {
  static auto step = (void(*)(const mjModel*, mjData*))dlsym(RTLD_NEXT, "mj_step");
  step(m, d);
  const char* path = getenv("IGRIS_GAIN_LOG");
  if (!path) return;
  static FILE* f = nullptr;
  static unsigned long count = 0;
  if (++count % 7) return;  // 300 Hz at the unchanged 2100 Hz physics timestep.
  if (!f) {
    f = fopen(path, "wb");
    if (!f) return;
    int h[] = {m->nq, m->nv, m->nu};
    fwrite(h, sizeof(int), 3, f);
    FILE* map = fopen((std::string(path) + ".map").c_str(), "w");
    for (int j=0; j<m->njnt; ++j)
      fprintf(map, "joint\t%s\t%d\t%d\n", mj_id2name(m,mjOBJ_JOINT,j), m->jnt_qposadr[j], m->jnt_dofadr[j]);
    for (int a=0; a<m->nu; ++a)
      fprintf(map, "actuator\t%s\t%d\t%.17g\t%.17g\n", mj_id2name(m,mjOBJ_ACTUATOR,a), a, m->actuator_ctrlrange[2*a],m->actuator_ctrlrange[2*a+1]);
    fprintf(map,"model\ttimestep\t%.17g\n",m->opt.timestep);
    fclose(map);
  }
  timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  std::vector<double> r = {t.tv_sec+t.tv_nsec*1e-9, d->time};
  r.insert(r.end(), d->qpos, d->qpos+m->nq);
  r.insert(r.end(), d->qvel, d->qvel+m->nv);
  r.insert(r.end(), d->ctrl, d->ctrl+m->nu);
  r.insert(r.end(), d->actuator_force, d->actuator_force+m->nu);
  for (const char* name : {"end_effector_left", "end_effector_right"}) {
    int b=mj_name2id(m,mjOBJ_BODY,name);
    r.insert(r.end(),d->xpos+3*b,d->xpos+3*b+3);
    r.insert(r.end(),d->xquat+4*b,d->xquat+4*b+4);
  }
  double self=0, min_distance=0;
  for (int i=0;i<d->ncon;++i) {
    auto& c=d->contact[i];
    if(c.geom[0]<0 || c.geom[1]<0) continue;
    if(m->geom_bodyid[c.geom[0]]>0 && m->geom_bodyid[c.geom[1]]>0) {
      mjtNum force[6]; mj_contactForce(m,d,i,force);
      if(force[0]>.5){self++; if(c.dist<min_distance)min_distance=c.dist;}
    }
  }
  r.insert(r.end(),{self,min_distance});
  fwrite(r.data(),sizeof(double),r.size(),f); fflush(f);
}
