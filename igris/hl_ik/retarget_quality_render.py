"""Side-by-side retargeting evidence at the same source phase (kinematic replay)."""
import argparse,json,os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from .common import OUTPUT,Rotation
from .retarget_quality import Refiner,fk
from . import retarget as gmr


def render(out):
    out=Path(out);report=json.loads((out/'comparison.json').read_text())
    with np.load(report['source']) as f:old={k:f[k] for k in f.files}
    with np.load(out/'clip.npz') as f:new={k:f[k] for k in f.files}
    refiner=Refiner();model=refiner.model;data=refiner.data
    rows={r['path']:r for r in map(json.loads,(OUTPUT/'data/retarget_results.jsonl').read_text().splitlines())}
    row=rows[report['source_path']];gmr.retargeter.human_scale_table={k:v*row['estimated_height_m']/1.8 for k,v in gmr.default_scale.items()}
    target_p,target_r,_=refiner.targets(old);torso=refiner.links.index('torso')
    for j in range(model.ngeom):
        if model.geom_group[j]==1:model.geom_rgba[j]=[.67,.71,.76,1.]
    model.vis.headlight.ambient[:]=.6;model.vis.headlight.diffuse[:]=.7
    model.vis.global_.offwidth=640;model.vis.global_.offheight=720
    option=mujoco.MjvOption();option.geomgroup[:]=0;option.geomgroup[1]=1
    camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.9];camera.distance=2.8;camera.azimuth=180;camera.elevation=-12
    renderer=mujoco.Renderer(model,720,640)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    def arrow(points,color,width=.006):
        for a,b in zip(points[:-1],points[1:]):
            geom=renderer.scene.geoms[renderer.scene.ngeom];renderer.scene.ngeom+=1
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.ones(3),np.zeros(3),np.eye(3).ravel(),np.array(color,np.float32))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,width,a,b)
    def frame(t):
        images=[];anchor_r=target_r[t,torso];anchor_p=target_p[t,torso]
        for title,clip,metrics in [('Original GMR',old,report['baseline']),('Refined GMR',new,report['refined'])]:
            q=clip['qpos'][t].astype(float).copy();q[:3]=anchor_r.T@(q[:3]-anchor_p)+[0,0,1.1]
            q[3:7]=Rotation.from_matrix(anchor_r.T@Rotation.from_quat(q[3:7],scalar_first=True).as_matrix()).as_quat(scalar_first=True)
            fk(model,data,q);renderer.update_scene(data,camera,option)
            for s in ('l','r'):
                indices=[refiner.links.index(s+'_'+n) for n in ('upper_arm','elbow_bracket','wrist_connector')]
                points=(target_p[t,indices]-anchor_p)@anchor_r+[0,0,1.1]
                arrow(points,[.2,1.,.35,1.])
            picture=Image.fromarray(renderer.render());draw=ImageDraw.Draw(picture)
            draw.rectangle((0,0,640,115),fill=(20,25,31));draw.text((16,10),title,font=font,fill='white')
            draw.text((16,40),f"All-frame mean position: {metrics['position_mean_m']*1000:.1f} mm",font=small,fill='white')
            draw.text((16,64),f"Self-collision frames: {metrics['rejection_counts']['self_collision']} / {metrics['frames']}",font=small,fill='white')
            draw.text((16,88),'Green: morphology target | Grey: robot FK',font=small,fill=(120,230,160))
            draw.rectangle((0,680,640,720),fill=(20,25,31));draw.text((16,689),f"Source phase {old['time'][t]:.2f}s | kinematic replay",font=small,fill='white')
            images.append(np.array(picture))
        return np.concatenate(images,axis=1)
    movie=imageio.get_writer(out/'comparison.mp4',fps=15,codec='libx264',quality=7)
    try:
        for t in range(0,len(old['time']),2):movie.append_data(frame(t))
        imageio.imwrite(out/'comparison.png',frame(len(old['time'])//2))
    finally:movie.close();renderer.close()
    print(out/'comparison.mp4')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);render(p.parse_args().out)
