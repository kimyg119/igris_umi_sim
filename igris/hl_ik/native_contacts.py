"""Opt-in compiled sphere selection with an exact-tie NumPy fallback."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile
import numpy as np


class NativeContacts:
    def __init__(self):
        source=Path(__file__).with_suffix('.cpp')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        cache=source.parents[2]/'.local/cache/hlik-native-contacts'
        cache.mkdir(parents=True,exist_ok=True)
        library=cache/(digest+'.so')
        if not library.exists():
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                compiled=Path(tmp)/'contacts.so'
                subprocess.run(['g++','-O2','-ffp-contract=off','-std=c++17','-shared','-fPIC',
                                str(source),'-o',str(compiled)],check=True)
                compiled.replace(library)
        self.library=ctypes.CDLL(str(library));self.function=self.library.select_contacts
        doubles=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        ints=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        self.function.argtypes=[doubles,doubles,ints,ctypes.c_int,ints,doubles,ctypes.c_int,
                                ctypes.c_int,ctypes.c_int,ints,doubles]
        self.function.restype=ctypes.c_int
        self.source_sha256=digest;self.tie_fallbacks=0

    def select(self,points,radii,pairs,thresholds,keep,stop_at_first=False):
        if keep<1:raise ValueError('Expected positive contact count')
        if not len(points) or len(points)!=len(radii):raise ValueError('Expected radii for every body')
        pairs=np.ascontiguousarray(pairs,dtype=np.int32).reshape(-1,2)
        thresholds=np.ascontiguousarray(thresholds,dtype=np.float64)
        if thresholds.shape!=(len(pairs),):raise ValueError('One threshold required per pair')
        if np.any(pairs<0) or np.any(pairs>=len(points)):raise ValueError('Invalid body pair')
        offsets=np.asarray([0,*np.cumsum([len(p) for p in points])],dtype=np.int32)
        xyz=np.ascontiguousarray(np.concatenate(points),dtype=np.float64)
        radius=np.ascontiguousarray(np.concatenate(radii),dtype=np.float64)
        if xyz.shape!=(len(radius),3) or any(len(p)!=len(r) for p,r in zip(points,radii)):
            raise ValueError('Sphere positions and radii differ')
        ids=np.empty((len(pairs)*keep,3),dtype=np.int32);gaps=np.empty(len(ids),dtype=np.float64)
        count=self.function(xyz,radius,offsets,len(points),pairs,thresholds,len(pairs),keep,int(stop_at_first),ids,gaps)
        if count<0:self.tie_fallbacks+=1;return None
        return [(int(k),int(i),int(j),float(gap)) for (k,i,j),gap in zip(ids[:count],gaps[:count])]
