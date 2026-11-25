#pragma once
#include <mujoco/mujoco.h>
#include <array>
#include <vector>
#include <cmath>
#include <cstring>

struct PayloadContactWrench {
    std::array<double,6> local{};
    std::array<double,3> force_world{}, torque_about_com_world{}, relative_velocity_world{};
    double cone_utilization=0;
};

// mj_contactForce acts on geom2; frame rows are axes in world coordinates.
// Read only: use the current solver result, never mj_forward on restored state.
inline PayloadContactWrench payload_contact_wrench(const mjModel* m,const mjData* d,int contact,int body) {
    const auto& ct=d->contact[contact];
    const int b1=m->geom_bodyid[ct.geom1],b2=m->geom_bodyid[ct.geom2];
    PayloadContactWrench out;
    if(body!=b1 && body!=b2)return out;
    const double sign=body==b2?1.:-1.;
    mj_contactForce(m,d,contact,out.local.data());
    std::vector<mjtNum> jp1(3*m->nv),jp2(3*m->nv);
    mj_jac(m,d,jp1.data(),nullptr,ct.pos,b1);
    mj_jac(m,d,jp2.data(),nullptr,ct.pos,b2);
    for(int k=0;k<3;k++) {
        for(int j=0;j<3;j++) {
            out.force_world[k]+=sign*ct.frame[3*j+k]*out.local[j];
            out.torque_about_com_world[k]+=sign*ct.frame[3*j+k]*out.local[3+j];
        }
        for(int v=0;v<m->nv;v++)out.relative_velocity_world[k]+=sign*(jp2[k*m->nv+v]-jp1[k*m->nv+v])*d->qvel[v];
    }
    mjtNum arm[3],moment[3];mju_sub3(arm,ct.pos,d->xipos+3*body);mju_cross(moment,arm,out.force_world.data());
    for(int k=0;k<3;k++)out.torque_about_com_world[k]+=moment[k];
    if(out.local[0]>0) {
        for(int k=1;k<ct.dim;k++)out.cone_utilization+=std::pow(out.local[k]/ct.friction[k-1],2);
        out.cone_utilization=std::sqrt(out.cone_utilization)/out.local[0];
    }
    return out;
}
