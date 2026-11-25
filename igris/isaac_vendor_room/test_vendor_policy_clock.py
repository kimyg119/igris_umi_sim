"""Only the audited policy scheduler may use native time; watchdogs stay real."""
from pathlib import Path
import os
import subprocess

import pytest

HERE = Path(__file__).resolve().parent


def test_native_elapsed_pause_reset_and_missing_clock(tmp_path):
    driver = tmp_path/'driver.cpp'
    driver.write_text('''
#define IGRIS_POLICY_CLOCK_TEST
#include "vendor_policy_clock.cpp"
#include <cassert>
int main() {
  NativeTimeline t;
  assert(t.sample(1000000000, false, 0.) == 1000000000);
  assert(t.sample(1050000000, false, 0.) == 1000000000);
  assert(t.sample(1100000000, true, 10.) == 1000000000);
  assert(t.sample(1200000000, true, 10.02) == 1020000000);
  assert(t.sample(1400000000, true, 10.02) == 1020000000);
  assert(t.sample(1500000000, false, 0.) == 1020000000);
  assert(t.sample(1600000000, true, 0.) == 1020000000);
  assert(t.sample(1800000000, true, .02) == 1040000000);
  for(auto offset: {0x5ab8aUL,0x5ac64UL,0x5ace9UL,0x5ad00UL})
    assert(policy_clock_callsite(offset));
  for(auto offset: {0UL,0x5ab89UL,0x5ac65UL,0x5ad01UL,0xa39e2UL})
    assert(!policy_clock_callsite(offset));
}''')
    subprocess.run(['g++','-std=c++17','-O2',f'-I{HERE}',str(driver),'-ldl','-o',str(tmp_path/'driver')],check=True)
    subprocess.run([str(tmp_path/'driver')],check=True)


def test_positive_process_and_mapped_native_snapshot(tmp_path):
    driver=tmp_path/'mapped.cpp'
    driver.write_text('''
#define IGRIS_POLICY_CLOCK_TEST
#include "vendor_policy_clock.cpp"
#include <cassert>
#include <limits>
int main(int argc,char**argv) {
  setenv("IGRIS_ROOM_POLICY_CLOCK_BINARY","/proc/self/exe",1);
  assert(selected_process());
  setenv("IGRIS_HLIK_CLOCK",argv[1],1);
  double sim=0;assert(!snapshot(sim)); // no file, no fallback sample
  int fd=open(argv[1],O_CREAT|O_RDWR,0600);assert(fd>=0);
  assert(!ftruncate(fd,sizeof(Clock)));
  auto c=static_cast<Clock*>(mmap(nullptr,sizeof(Clock),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0));
  assert(c!=MAP_FAILED);timespec now{};clock_gettime(CLOCK_MONOTONIC,&now);
  c->monotonic=now.tv_sec+now.tv_nsec*1e-9;c->sim=4.2;c->sequence=2;
  assert(snapshot(sim) && sim==4.2);
  c->sequence=3;assert(!snapshot(sim));
  c->sequence=4;c->sim=std::numeric_limits<double>::quiet_NaN();assert(!snapshot(sim));
  c->sim=4.3;c->monotonic-=1.;assert(!snapshot(sim));
  munmap(c,sizeof(Clock));close(fd);
}''')
    subprocess.run(['g++','-std=c++17','-O2',f'-I{HERE}',str(driver),'-ldl','-o',str(tmp_path/'mapped')],check=True)
    subprocess.run([str(tmp_path/'mapped'),str(tmp_path/'clock.bin')],check=True)


def test_pinned_binary_and_unrelated_process_clock(tmp_path):
    from vendor_policy_clock import build, BINARY_SHA256, binary_manifest
    audit=binary_manifest()
    assert audit['sha256']==BINARY_SHA256
    assert audit['return_offsets']==[0x5ab8a,0x5ac64,0x5ace9,0x5ad00]
    library=build(tmp_path)
    # An unrelated executable must retain wall time even with the library
    # and a valid native-clock file in its environment.
    (tmp_path/'plain.cpp').write_text('''
#include <chrono>
#include <thread>
#include <cstdio>
int main(){auto begin=std::chrono::steady_clock::now();
std::this_thread::sleep_for(std::chrono::milliseconds(40));
printf("%.9f",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());}
''')
    subprocess.run(['g++','-std=c++17',str(tmp_path/'plain.cpp'),'-o',str(tmp_path/'plain')],check=True)
    import struct,time
    clock=tmp_path/'native_clock.bin'
    clock.write_bytes(struct.pack('Qddd',2,time.time(),time.monotonic(),10.))
    env=os.environ.copy()
    env.update(LD_PRELOAD=str(library),IGRIS_HLIK_CLOCK=str(clock),
               IGRIS_ROOM_POLICY_CLOCK_BINARY=audit['binary'],IGRIS_ROOM_OUTPUT=str(tmp_path))
    elapsed=float(subprocess.check_output([str(tmp_path/'plain')],env=env))
    assert .035<=elapsed<.5
    assert not (tmp_path/'vendor_policy_clock.jsonl').exists()


def test_wrong_binary_refused(tmp_path):
    from vendor_policy_clock import binary_manifest
    fake=tmp_path/'sim';fake.write_bytes(b'unrelated')
    with pytest.raises(ValueError,match='fingerprint'):
        binary_manifest(fake)
