"""Columns of the LeRobot export (lerobot_export.py); writing the dataset itself needs LeRobot and a GPU context."""
import numpy as np

from igris.umi_walklow_replay import ee_features, lerobot_export
from igris.umi_walklow_replay.test_ee_dataset import fake_run
from igris.umi_walklow_replay import ee_dataset


def test_columns_match_their_declared_shapes_and_the_relative_action_is_the_one_step_one(tmp_path):
    out, _ = fake_run(tmp_path)
    data = ee_dataset.frames(out)
    columns = lerobot_export.rows(data)
    declared = lerobot_export.features()
    count = len(data['t'])
    for key, values in columns.items():
        assert values.shape == (count, *declared[key]['shape']) and str(values.dtype) == declared[key]['dtype'], key
        if declared[key]['names'] is not None:
            assert len(declared[key]['names']) == declared[key]['shape'][0], key
    frame = 5
    wanted = ee_features.relative_actions(data['ee_room'][frame], data['q_measured'][frame, :3],
                                          data['target_room'][frame:frame + 1], data['waist_target'][frame:frame + 1])[0]
    assert np.allclose(columns['action.relative'][frame], wanted, atol=1e-6)
    # The stored hands and targets give the same relative action back (what a chunk is made from).
    hands = np.stack([ee_features.from9(columns['observation.ee_room'][frame, :9]), ee_features.from9(columns['observation.ee_room'][frame, 9:])])
    target = np.stack([ee_features.from9(columns['action.room'][frame, :9]), ee_features.from9(columns['action.room'][frame, 9:])])[None]
    again = ee_features.relative_actions(hands, data['q_measured'][frame, :3], target, columns['action'][frame:frame + 1, 18:])[0]
    assert np.allclose(again, wanted, atol=1e-5)
    assert set(declared) - set(columns) == {f'observation.images.{name}' for name in ('head_cam', 'left_cam', 'right_cam')}
