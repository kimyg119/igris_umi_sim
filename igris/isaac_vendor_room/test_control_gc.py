"""Accumulating command logs must not trigger cyclic collection in the control loop."""
import gc
import pytest
from control_gc import defer_cyclic_gc


def test_control_ownership_closes_before_gc_restores_on_error():
    events=[]
    def close():events.append(('closed',gc.isenabled()))
    with pytest.raises(RuntimeError,match='control failed'):
        with defer_cyclic_gc(before_restore=close):
            raise RuntimeError('control failed')
    assert events==[('closed',False)]
    assert gc.isenabled()


def test_gc_restored_even_if_ownership_cleanup_fails():
    def close():raise ValueError('cleanup failed')
    with pytest.raises(ValueError,match='cleanup failed'):
        with defer_cyclic_gc(before_restore=close):pass
    assert gc.isenabled()


def check():
    prior=gc.isenabled();thresholds=gc.get_threshold();events=[];active=False
    def observe(phase,info):
        if active and phase=='start':events.append(info['generation'])
    gc.enable();gc.set_threshold(20,2,2);gc.callbacks.append(observe)
    try:
        with defer_cyclic_gc():
            active=True;rows=[{'sample':[float(i),i]} for i in range(10000)];active=False
        assert not events, f'Automatic GC interrupted control: {events}'
        assert len(rows)==10000 and gc.isenabled()
        try:
            with defer_cyclic_gc():raise RuntimeError('Control guard failed')
        except RuntimeError:pass
        assert gc.isenabled(), 'GC was not restored after control failure'
        gc.disable()
        with defer_cyclic_gc():pass
        assert not gc.isenabled(), 'Caller-disabled GC was enabled'
    finally:
        active=False;gc.callbacks.remove(observe);gc.set_threshold(*thresholds)
        (gc.enable if prior else gc.disable)()
    print('PASS: growing logs do not interrupt bounded control; GC state restored on success/error')

if __name__=='__main__':check()
