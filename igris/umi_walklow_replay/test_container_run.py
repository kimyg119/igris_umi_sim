"""The in-container launcher (container_run.py): what can be checked without a simulator."""
import os

from igris.umi_walklow_replay import container_run


def test_physics_cpu_is_kept_free_of_everything_else(monkeypatch):
    allowed = sorted(os.sched_getaffinity(0))
    monkeypatch.setattr(container_run, 'OTHERS', [])
    cpu = container_run.reserve()
    assert cpu == allowed[0]
    if len(allowed) > 1:
        assert cpu not in container_run.OTHERS and set(container_run.OTHERS) < set(allowed)
        command = container_run.elsewhere(['bash', '-lc', 'true'])
        assert command[:2] == ['taskset', '-c'] and command[3:] == ['bash', '-lc', 'true']
        assert str(cpu) not in command[2].split(',')


def test_without_a_reserved_cpu_commands_run_as_they_are(monkeypatch):
    monkeypatch.setattr(container_run, 'OTHERS', [])
    assert container_run.elsewhere(['echo', 'x']) == ['echo', 'x']


def test_outside_a_private_container_nothing_is_started(tmp_path):
    import pytest
    if container_run.isolated():
        pytest.skip('running inside an isolated container')
    with pytest.raises(RuntimeError):
        container_run.start(tmp_path)
