#pragma once
// Classify existing solved payload wrenches by explicit room-body membership.
// No contact eligibility, force, model or state is modified.
#include <mujoco/mujoco.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct SupportForceGroups {
 bool initialized=false,configured=false;
 const mjModel* bound_model=nullptr;
 std::vector<std::string> source_names,destination_names;
 std::vector<bool> source,destination;
 [[noreturn]] static void fail(const std::string& reason) {
  std::fprintf(stderr,"support force selector: %s\n",reason.c_str());std::abort();
 }
 static std::vector<std::string> read(const char* path) {
  std::ifstream file(path);if(!file)fail("cannot read body list");
  std::vector<std::string> names;std::string line;
  while(std::getline(file,line)) {
   const auto first=line.find_first_not_of(" \t\r");if(first==std::string::npos)continue;
   names.push_back(line.substr(first,line.find_last_not_of(" \t\r")-first+1));
  }
  if(file.bad() || names.empty())fail("empty or unreadable body list");
  return names;
 }
 static void quoted(FILE* f,const std::string& value) {
  std::fputc('"',f);
  for(unsigned char c:value) {
   if(c=='"'||c=='\\'){std::fputc('\\',f);std::fputc(c,f);}
   else if(c<0x20)std::fprintf(f,"\\u%04x",c);
   else std::fputc(c,f);
  }
  std::fputc('"',f);
 }
 static void names_json(FILE* f,const std::vector<std::string>& names) {
  std::fputc('[',f);for(size_t i=0;i<names.size();i++){if(i)std::fputc(',',f);quoted(f,names[i]);}std::fputc(']',f);
 }
 void prepare(const mjModel* m,const char* payload,const char* directory) {
  if(!initialized) {
   initialized=true;
   const char* s=std::getenv("IGRIS_ROOM_SOURCE_SUPPORT_BODIES_FILE");
   const char* t=std::getenv("IGRIS_ROOM_DESTINATION_SUPPORT_BODIES_FILE");
   if(bool(s)!=bool(t))fail("source and destination lists must be supplied together");
   configured=s&&t;
   if(configured){source_names=read(s);destination_names=read(t);}
  }
  if(bound_model==m)return;
  source.assign(m->nbody,false);destination.assign(m->nbody,false);
  int payload_id=-1;
  if(configured) {
   if(!payload || !*payload || (payload_id=mj_name2id(m,mjOBJ_BODY,payload))<1)fail("missing or unknown payload body");
  }
  auto bind=[&](const std::vector<std::string>& names,std::vector<bool>& selected) {
   for(const auto& name:names) {
    int b=mj_name2id(m,mjOBJ_BODY,name.c_str());
    if(name.compare(0,5,"room_") || b<1)fail("unknown or non-room body: "+name);
    if(b==payload_id)fail("payload cannot support itself");
    if(m->body_jntnum[b]!=1 || m->jnt_type[m->body_jntadr[b]]!=mjJNT_FREE)fail("support body is not free: "+name);
    if(selected[b])fail("duplicate support body: "+name);
    selected[b]=true;
   }
  };
  bind(source_names,source);bind(destination_names,destination);
  for(int b=0;b<m->nbody;b++)if(source[b]&&destination[b])fail("source and destination groups overlap");
  FILE* f=std::fopen((std::string(directory)+"/support_force_groups.json").c_str(),"w");
  if(!f)fail("cannot write group provenance");
  std::fprintf(f,"{\"schema\":\"native_payload_support_groups/v1\",\"configured\":%s,\"source_bodies\":",configured?"true":"false");
  names_json(f,source_names);std::fputs(",\"destination_bodies\":",f);names_json(f,destination_names);
  std::fputs(",\"force_semantics\":\"Signed world-Z force on payload from actual solved contacts with exactly listed bodies; excludes robot forearms and unlisted room bodies\",\"legacy_total_unchanged\":true}\n",f);
  if(std::fclose(f))fail("cannot close group provenance");
  bound_model=m;
 }
};
