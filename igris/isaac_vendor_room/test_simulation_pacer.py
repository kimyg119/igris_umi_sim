"""Native pacing must preserve every integrated state and bound wall progress."""
from pathlib import Path
import os
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def test_replay_pacing_waits_for_selected_phase_and_is_one_shot(tmp_path):
    from run import request_replay_pacing
    assert not request_replay_pacing(tmp_path, .5, 0., 90.)
    assert not request_replay_pacing(tmp_path, .5, 89.999, 90.)
    assert not (tmp_path/'pacing.request').exists()
    assert request_replay_pacing(tmp_path, .5, 90., 90.)
    stamp=(tmp_path/'pacing.request').stat().st_mtime_ns
    assert not request_replay_pacing(tmp_path, .5, 91., 90.)
    assert (tmp_path/'pacing.request').stat().st_mtime_ns==stamp


def test_replay_pacing_default_and_disabled_behavior(tmp_path):
    from run import request_replay_pacing
    assert not request_replay_pacing(tmp_path, None, 100., 0.)
    assert not (tmp_path/'pacing.request').exists()
    assert request_replay_pacing(tmp_path, .5, 0., 0.)


def check():
    inc=ROOT/'igris/ws/install/igris_c_mujoco/include'
    lib=ROOT/'igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco'
    with tempfile.TemporaryDirectory(prefix='igris-pacer-') as tmp:
        p=Path(tmp)
        (p/'model.xml').write_text('''<mujoco><option timestep="0.001"/>
          <worldbody><body pos="0 0 1"><joint name="slide" type="slide" axis="0 0 1"/>
          <geom type="sphere" size=".05" mass="1"/></body></worldbody>
          <actuator><motor joint="slide"/></actuator></mujoco>''')
        (p/'driver.cpp').write_text('''#include <mujoco/mujoco.h>
          #include <cstdio>
          #include <vector>
          int main(int argc,char** argv) {
            char error[1024]{};auto m=mj_loadXML(argv[1],nullptr,error,1024);
            if(!m)return 2;auto d=mj_makeData(m);
            std::vector<double> state(mj_stateSize(m,mjSTATE_INTEGRATION));
            for(int i=0;i<200;i++) {
              if(i==100 && argc>2) {auto f=fopen(argv[2],"w");if(!f)return 3;fclose(f);}
              if(i==100 && argc>3) {auto f=fopen(argv[3],"w");if(!f)return 4;fclose(f);}
              d->ctrl[0]=(i%17)*.1;mj_step(m,d);
              mj_getState(m,d,state.data(),mjSTATE_INTEGRATION);
              fwrite(state.data(),sizeof(double),state.size(),stdout);
            }
            mj_deleteData(d);mj_deleteModel(m);
          }''')
        subprocess.run(['g++','-O2',f'-I{inc}',str(p/'driver.cpp'),f'-L{lib}',
                        f'-Wl,-rpath,{lib}','-l:libmujoco.so.3.2.7','-o',str(p/'driver')],check=True)
        subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC',f'-I{inc}',
                        str(HERE/'simulation_pacer.cpp'),'-ldl','-o',str(p/'pacer.so')],check=True)
        command=[str(p/'driver'),str(p/'model.xml')]
        baseline=subprocess.check_output(command)
        env=os.environ.copy();env['LD_PRELOAD']=str(p/'pacer.so')
        assert subprocess.check_output(command,env=env)==baseline
        elapsed={}
        for rate in (1.,.25):
            env['IGRIS_ROOM_MAX_REALTIME_FACTOR']=str(rate)
            start=time.monotonic();actual=subprocess.check_output(command,env=env)
            elapsed[rate]=time.monotonic()-start
            assert actual==baseline, 'Pacing changed a native integration state'
            assert elapsed[rate]>=.199/rate-.003, (rate,elapsed)
        env['IGRIS_ROOM_PACER_START']=str(p/'start.request')
        start=time.monotonic()
        assert subprocess.check_output(command,env=env)==baseline
        assert time.monotonic()-start<.4, 'Pacing started before preparation acknowledged it'
        start=time.monotonic()
        assert subprocess.check_output(command+[str(p/'start.request')],env=env)==baseline
        activated_elapsed=time.monotonic()-start
        assert .393<=activated_elapsed<.75, activated_elapsed
        env.pop('IGRIS_ROOM_PACER_START')
        # A loaded handoff may reduce wall rate once, without resetting any
        # integrated state or changing the preceding acquisition dynamics.
        import json
        env['IGRIS_ROOM_MAX_REALTIME_FACTOR']='.5'
        env['IGRIS_ROOM_SLOW_REALTIME_FACTOR']='.1'
        env['IGRIS_ROOM_PACER_SLOWDOWN']=str(p/'slow.request')
        start=time.monotonic()
        actual=subprocess.check_output(command+[str(p/'unused.start'),str(p/'slow.request')],env=env)
        slow_elapsed=time.monotonic()-start
        assert actual==baseline, 'Slowdown changed a native integration state'
        assert 1.18<=slow_elapsed<2., slow_elapsed
        ack=json.loads((p/'slow.request.ack').read_text())
        assert ack['previous_rate']==.5 and ack['rate']==.1
        assert abs(ack['native_time_s']-.1)<1e-12
        for bad_slow in ('0','nan','.6'):
            env['IGRIS_ROOM_SLOW_REALTIME_FACTOR']=bad_slow
            result=subprocess.run(command,env=env,capture_output=True)
            assert result.returncode!=0 and not result.stdout
        env.pop('IGRIS_ROOM_SLOW_REALTIME_FACTOR')
        env.pop('IGRIS_ROOM_PACER_SLOWDOWN')
        for invalid in ('0','-1','nan','inf','1.1','0.25junk'):
            env['IGRIS_ROOM_MAX_REALTIME_FACTOR']=invalid
            result=subprocess.run(command,env=env,capture_output=True)
            assert result.returncode!=0 and not result.stdout, invalid
        print('PASS: 200 native states byte-identical; wall-rate cap enforced; invalid values rejected',elapsed)


if __name__=='__main__':check()
