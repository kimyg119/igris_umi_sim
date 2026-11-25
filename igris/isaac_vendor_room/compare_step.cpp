// Compare dense/sparse arithmetic on one captured, complete vendor state.
// No controller or hardware connection. Each trial restores the same input.
#include <mujoco/mujoco.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

static double clock_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double difference(const std::vector<mjtNum>& a,const mjtNum* b) {
    double result=0;
    for(size_t i=0;i<a.size();i++) {
        if(!std::isfinite(a[i]) || !std::isfinite(b[i]))return std::numeric_limits<double>::infinity();
        result=std::fmax(result,std::abs(a[i]-b[i]));
    }
    return result;
}
int main(int argc,char** argv) {
    if(argc!=4) {std::fprintf(stderr,"usage: compare_step scene.xml state.bin output.json\n");return 2;}
    char error[4096]{};auto m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
    if(!m) {std::fprintf(stderr,"%s\n",error);return 1;}
    std::vector<mjtNum> input(mj_stateSize(m,mjSTATE_INTEGRATION));
    FILE* source=std::fopen(argv[2],"rb");
    if(!source || std::fread(input.data(),sizeof(mjtNum),input.size(),source)!=input.size() || std::fgetc(source)!=EOF)return 1;
    std::fclose(source);
    FILE* output=std::fopen(argv[3],"w");if(!output)return 1;
    std::vector<mjtNum> q,v,acc,force,contact;
    std::fprintf(output,"{\"mujoco_version\":\"%s\",\"same_integration_state_and_input\":true,\"trials_per_case\":200,\"required_step_ms\":%.17g,\"cases\":[",mj_versionString(),1000*m->opt.timestep);
    int case_index=0;bool ok=true;
    for(int jacobian:{mjJAC_DENSE,mjJAC_SPARSE})for(int threads:{0,2,4}) {
        if(jacobian==mjJAC_DENSE && threads)continue;
        m->opt.jacobian=jacobian;
        auto d=mj_makeData(m);auto pool=threads?mju_threadPoolCreate(threads):nullptr;
        if(pool)mju_bindThreadPool(d,pool);
        double elapsed=0;
        for(int i=0;i<210;i++) {
            mj_setState(m,d,input.data(),mjSTATE_INTEGRATION);
            double begin=clock_seconds();mj_step(m,d);
            if(i>=10)elapsed+=clock_seconds()-begin;
        }
        std::vector<mjtNum> cf;
        for(int i=0;i<d->ncon;i++) {
            mjtNum wrench[6];mj_contactForce(m,d,i,wrench);
            cf.insert(cf.end(),wrench,wrench+6);
        }
        if(!case_index) {
            q.assign(d->qpos,d->qpos+m->nq);v.assign(d->qvel,d->qvel+m->nv);
            acc.assign(d->qacc,d->qacc+m->nv);force.assign(d->qfrc_constraint,d->qfrc_constraint+m->nv);contact=cf;
        }
        double eq=difference(q,d->qpos),ev=difference(v,d->qvel),ea=difference(acc,d->qacc),ef=difference(force,d->qfrc_constraint);
        double ec=contact.size()==cf.size()?difference(contact,cf.data()):-1;
        bool parity=eq<1e-9 && ev<1e-7 && ea<1e-4 && ef<1e-4 && ec>=0 && ec<1e-4;
        ok&=parity;
        std::fprintf(output,"%s{\"jacobian\":\"%s\",\"threads\":%d,\"mean_step_ms\":%.17g,\"max_qpos_delta\":%.17g,\"max_qvel_delta\":%.17g,\"max_qacc_delta\":%.17g,\"max_constraint_force_delta\":%.17g,\"max_contact_wrench_delta\":%.17g,\"ncon\":%d,\"nefc\":%d,\"numerical_check_pass\":%s}",case_index?",":"",jacobian==mjJAC_DENSE?"dense":"sparse",threads,1000*elapsed/200,eq,ev,ea,ef,ec,d->ncon,d->nefc,parity?"true":"false");
        std::printf("%s threads=%d step_ms=%.6f qpos_delta=%.3g qacc_delta=%.3g contact_delta=%.3g\n",jacobian==mjJAC_DENSE?"dense":"sparse",threads,1000*elapsed/200,eq,ea,ec);
        mj_deleteData(d);if(pool)mju_threadPoolDestroy(pool);case_index++;
    }
    std::fprintf(output,"],\"numerical_check_pass\":%s,\"limits\":\"Single captured-state comparison, not bitwise or long-horizon equivalence\"}\n",ok?"true":"false");
    std::fclose(output);mj_deleteModel(m);return ok?0:1;
}
