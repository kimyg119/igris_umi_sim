"""Read the consistent latest physics clock, independently of saved video logs."""
import struct
import time
import math


def read_clock(path,*,now=None,max_age=.15):
    with open(path,'rb',buffering=0) as f:
        for _ in range(8):
            f.seek(0);raw=f.read(32);f.seek(0);again=f.read(8)
            if len(raw)!=32 or len(again)!=8:continue
            seq,wall,monotonic,sim=struct.unpack('Qddd',raw)
            if seq and not seq%2 and seq==struct.unpack('Q',again)[0]:
                if not all(math.isfinite(v) for v in (wall,monotonic,sim)) or sim<0:
                    raise RuntimeError('invalid native physics clock')
                age=time.monotonic()-monotonic if now is None else now-wall
                if not 0<=age<=max_age:raise RuntimeError('stale native physics clock')
                return dict(sequence=seq,wall=wall,monotonic=monotonic,sim=sim)
    raise RuntimeError('incomplete native physics clock update')
