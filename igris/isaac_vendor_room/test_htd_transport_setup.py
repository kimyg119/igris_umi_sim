import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
import pytest
import numpy as np
import htd_loaded_setup as setup
from htd_loaded_runtime import LoadedRuntime
from test_htd_loaded_runtime import Node,Service,arguments
from test_htd_candidate_package import package


def test_transport_without_full_explicit_scope_rejects_before_files():
    with pytest.raises(ValueError,match='transport'):
        setup.prepare(Path('/unused'),enabled=True,transport=True)


@pytest.mark.parametrize('candidate',[False,True])
def test_transport_setup_and_worker_share_destination_duration_and_feedback(candidate,package):
    with tempfile.TemporaryDirectory(prefix='htd_transport_test_',dir=setup.ROOT/'outputs') as directory:
        trial=Path(directory)
        for name in ('manifest.json','robot.yaml','scene.xml','vendor.xml'):
            shutil.copyfile(setup.P/'transfer_trial76/run'/name,trial/name)
        path=trial/'manifest.json';manifest=json.loads(path.read_text())
        manifest['walking_transfer'].update(heading_preserving_route=True,xy_first_lowering=True,
            world_position_lowering=True,torso_relative_lowering=True)
        path.write_text(json.dumps(manifest))
        evidence=setup.prepare(trial,enabled=True,current_grasp_reference=True,command_observer=True,
            wrist_mode_entry=True,goal_hold=True,contact_grace=True,transport=True,
            policy_package=package[0] if candidate else None)
        scope=evidence['manifest_scope']
        assert scope['scope']=='loaded_transport' and scope['duration_s']==180.
        assert scope['transport']['target_payload_xyz']==manifest['walking_transfer']['target_payload_xyz']
        assert scope['transport']['primary_qp_start']=='cold'
        assert scope['transport']['primary_qp_iteration_limits']=={'outer':10,'inner':10}
        source=setup.ROOT/'igris/hl_ik/pipeline.py'
        assert scope['setup_sha256'][str(source)]==setup.sha(source)
        assert scope['goal_hold']['config']['departure_radius']==.03
        (trial/'runtime.json').write_text(json.dumps(dict(network='none',ipc='private',hardware_executed=False,
                                                       simulation_rate_limit=.5)))
        node=Node()
        obj=LoadedRuntime(trial,node,services={'mode':Service,'sim':Service},spin=lambda:None)
        assert obj.coordinator.transport and obj.coordinator.duration_s==180.
        assert obj.coordinator.goal_hold.contract==scope['goal_hold']
        assert obj.coordinator.policy._step_index==0 and node.requests==[]
        q=obj.coordinator.hold_q.copy()
        obj._wall=lambda:2000.;obj._monotonic=lambda:1000.;obj._clock=lambda:dict(sim=100.)
        obj._output=lambda:dict(sim=100.,monotonic=1000.,sequence=2,
            q_output=q.copy(),dq_output=np.zeros(31),q_encoder=q.copy(),dq_encoder=np.zeros(31))
        args=arguments(obj)
        args.update(torso_index=1,payload_index=2,hand_indices=[3,4],control_time_s=10.01,
            applied_segment=dict(start_time=10.,end_time=10.+1/30,accepted=True,prior_status='used',
                walking_phase='lower',reference_control_time_s=9.9,control_points=[q[15:29].tolist()]*4))
        args['poses']=np.array([[0.,0.,.9,1.,0.,0.,0.],[-.05,0.,.94,1.,0.,0.,0.],
                               [.25,-.2,.758,1.,0.,0.,0.],[.25,-.3,.758,1.,0.,0.,0.],[.25,-.1,.758,1.,0.,0.,0.]])
        args['forces'].update(destination_support_force_world_z_N=.981,sim=100.)
        args['walking_state'].update(phase='open',target_payload_xyz=scope['transport']['target_payload_xyz'],done=False)
        snapshot=obj.snapshot(**args)
        assert snapshot.transport_phase=='lower' and snapshot.phase_target_valid
        assert snapshot.destination_support_z_N==.981 and not snapshot.placement_done
        np.testing.assert_array_equal(snapshot.placement_target_xyz,scope['transport']['target_payload_xyz'])
        obj.wrist_entry_started=True
        obj.wrist_entry.ready_for_low=lambda s:True
        assert not obj.entry_ready(replace(snapshot,transport_phase='carry',phase_target_valid=False))
        assert obj.entry_ready(replace(snapshot,transport_phase='carry',phase_target_valid=True))
        # A scene target changed after setup cannot change the loaded route.
        manifest=json.loads(path.read_text());manifest['walking_transfer']['target_payload_xyz'][0]+=.01
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError,match='transport|Transport'):
            LoadedRuntime(trial,Node(),services={'mode':Service,'sim':Service})
