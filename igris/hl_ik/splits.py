"""Curated source-identity split, independent of retargeted motion values."""
from collections import Counter
from .common import *

def curated_manifest():
    rows=json.loads((OUTPUT/'data/source_manifest.json').read_text())['files']
    for r in rows:
        parts=Path(r['path']).parts
        if r['corpus']=='EyesJapanDataset':r['group']=r['corpus']+'/'+parts[2]
    groups=sorted({r['group'] for r in rows if r['corpus']=='EyesJapanDataset'},key=lambda g:hashlib.sha256(g.encode()).hexdigest())
    hold=max(1,round(len(groups)*.1))
    mapping={g:'test' if i<hold else 'val' if i<2*hold else 'train' for i,g in enumerate(groups)}
    shared={Path(r['path']).parts[1]:r['split'] for r in rows if r['corpus']=='DFaust'}
    for r in rows:
        if r['corpus']=='EyesJapanDataset':r['split']=mapping[r['group']]
        person=Path(r['path']).parts[1]
        if r['corpus'] in ('DFaust','MoSh') and person in shared:
            r['group']='shared_smpl/'+person;r['split']=shared[person]
    return rows

def main():
    rows=curated_manifest()
    groups={s:{r['group'] for r in rows if r['split']==s} for s in ('train','val','test')}
    assert not(groups['train']&groups['test'] or groups['train']&groups['val'] or groups['val']&groups['test'])
    save_json(OUTPUT/'data/split_manifest.json',dict(files=rows,source_manifest_sha256=sha256(OUTPUT/'data/source_manifest.json'),
        policy='provided subject IDs held out per corpus; EyesJapan wrapper removed; identical DFaust/MoSh 500xx IDs conservatively share the DFaust split',
        limitations=['CMU states that one person may have several subject numbers; undisclosed aliases cannot be certified person-disjoint',
            'corpora with fewer than three subject groups remain training only; no test metric is claimed for those corpora'],
        counts={s:dict(Counter(r['corpus'] for r in rows if r['split']==s and r['status']=='pending')) for s in groups}))

if __name__=='__main__':main()
