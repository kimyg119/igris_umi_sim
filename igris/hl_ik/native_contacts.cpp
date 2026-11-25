// Exact sphere-gap selection; no geometry, threshold, or solver approximation.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

struct Hit {double gap;int i,j;};

extern "C" int select_contacts(const double* points,const double* radii,const int* offsets,
        int bodies,const int* pairs,const double* thresholds,int pair_count,int keep,
        int stop_first,int* indices,double* gaps) {
    std::vector<std::array<double,6>> boxes(bodies);
    for(int b=0;b<bodies;b++) {
        auto& box=boxes[b];box={INFINITY,INFINITY,INFINITY,-INFINITY,-INFINITY,-INFINITY};
        for(int i=offsets[b];i<offsets[b+1];i++)for(int axis=0;axis<3;axis++) {
            box[axis]=std::min(box[axis],points[3*i+axis]-radii[i]);
            box[axis+3]=std::max(box[axis+3],points[3*i+axis]+radii[i]);
        }
    }
    int count=0;
    for(int k=0;k<pair_count;k++) {
        int a=pairs[2*k],b=pairs[2*k+1];double threshold=thresholds[k];
        const auto& ba=boxes[a];const auto& bb=boxes[b];
        bool disjoint=false;
        for(int axis=0;axis<3;axis++)if(ba[axis]>bb[axis+3]+threshold || bb[axis]>ba[axis+3]+threshold)disjoint=true;
        if(disjoint)continue;
        std::vector<int> ia,ib;
        for(int side=0;side<2;side++) {
            int body=side?b:a;const auto& other=side?ba:bb;
            auto& selected=side?ib:ia;
            for(int i=offsets[body];i<offsets[body+1];i++) {
                bool overlap=true;
                for(int axis=0;axis<3;axis++)if(points[3*i+axis]+radii[i]+threshold<other[axis] ||
                        points[3*i+axis]-radii[i]-threshold>other[axis+3])overlap=false;
                if(overlap)selected.push_back(i);
            }
        }
        std::vector<Hit> hits;
        for(int i:ia)for(int j:ib) {
            double dx=points[3*i]-points[3*j],dy=points[3*i+1]-points[3*j+1],dz=points[3*i+2]-points[3*j+2];
            double gap=std::sqrt(dx*dx+dy*dy+dz*dz)-radii[i]-radii[j];
            if(gap<threshold)hits.push_back({gap,i-offsets[a],j-offsets[b]});
        }
        int n=std::min(keep,static_cast<int>(hits.size()));
        if(!n)continue;
        std::partial_sort(hits.begin(),hits.begin()+n,hits.end(),[](auto& x,auto& y){return x.gap<y.gap;});
        // NumPy's historical quicksort tie ordering is part of the interface.
        // Delegate exact ties to that implementation rather than invent order.
        for(int i=1;i<n;i++)if(hits[i].gap==hits[i-1].gap)return -1;
        for(size_t i=n;i<hits.size();i++)if(hits[i].gap==hits[n-1].gap)return -1;
        for(int i=0;i<n;i++) {
            indices[3*count]=k;indices[3*count+1]=hits[i].i;indices[3*count+2]=hits[i].j;
            gaps[count++]=hits[i].gap;
            if(stop_first)return count;
        }
    }
    return count;
}
