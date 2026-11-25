"""Plots and video from saved predictions and measured native physics."""
import argparse,json,sys,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def plots():
    out=OUTPUT/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    colors={'paper':'#2878b5','full':'#e17827'}
    for scope in colors:
        for seed in (42,123,789,1024):
            h=json.loads((OUTPUT/'training'/scope/f'seed_{seed}/history.json').read_text())
            for ax,key in zip(axes,('position_mean_m','orientation_mean_rad')):
                scale=1000 if key.endswith('_m') else 180/np.pi
                ax.plot([r['epoch'] for r in h],[r[key]*scale for r in h],color=colors[scope],alpha=.6,label=scope if seed==42 else None)
    for ax,label in zip(axes,('Validation elbow position (mm)','Validation elbow orientation (deg)')):
        ax.set(xlabel='Epoch',ylabel=label,yscale='log');ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'training_curves.png',dpi=170);plt.close(fig)
    rows=json.loads((OUTPUT/'evaluation/common_prediction.json').read_text())['results']
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,key,baseline,scale,label in zip(axes,('position_mean_m','orientation_mean_rad'),('persistence_position_mean_m','persistence_orientation_mean_rad'),(1000,180/np.pi),('Elbow position MAE (mm)','Elbow orientation MAE (deg)')):
        for i,scope in enumerate(colors):
            vals=np.array([r[key] for r in rows if r['scope']==scope])*scale
            ax.bar(i,vals.mean(),yerr=vals.std(ddof=1),capsize=5,color=colors[scope]);ax.scatter(np.full(4,i),vals,c='black',s=12)
        ax.axhline(rows[0][baseline]*scale,color='gray',linestyle='--',label='Previous elbow pose')
        ax.set(xticks=[0,1],xticklabels=['Paper corpora','All corpora'],ylabel=label,title='Same held-out test windows');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'common_test.png',dpi=180);plt.close(fig)
    quality=json.loads((OUTPUT/'evaluation/data_quality.json').read_text());cs=quality['corpora']
    names=list(cs);total=np.array([cs[n]['frames'] for n in names]);accepted=np.array([cs[n]['accepted_frames'] for n in names])
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained');x=np.arange(len(names))
    ax.bar(x,total/1000,color='#dddddd',label='Retargeted frames');ax.bar(x,accepted/1000,color='#30897c',label='All quality gates passed')
    ax.set(xticks=x,xticklabels=names,ylabel='Frames (thousands)',title='Entire local AMASS population; rejected frames never bridge training windows')
    ax.tick_params(axis='x',rotation=65);ax.legend();fig.savefig(out/'data_quality.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    for name,label,color in [('numerical_bounded','Numerical','#777777'),('paper_guarded','Paper corpora',colors['paper']),('full_guarded','All corpora',colors['full'])]:
        f=np.load(OUTPUT/'box_plans'/name/'ik.npz');s=f['statistics'];t=f['time']
        axes[0].plot(t,s[:,0]*1000,label=label,color=color);axes[1].plot(t,np.rad2deg(s[:,1]),color=color)
    axes[0].axhline(15,color='red',ls='--',lw=1);axes[1].axhline(2,color='red',ls='--',lw=1)
    axes[0].set(ylabel='Max L/R EE error (mm)',title='1 kg box path: own generated IK history at 30 Hz');axes[0].legend()
    axes[1].set(xlabel='Motion time (s)',ylabel='Max L/R orientation (deg)')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'box_ik_tracking.png',dpi=180);plt.close(fig)

def video(name,plan,label,speed=3.,fps=15):
    if not 0<speed or not 1<=fps<=120:raise ValueError('positive playback speed and fps in1..120 required')
    import mujoco
    from PIL import Image,ImageDraw,ImageFont
    sys.path.insert(0,str(IGRIS/'sim_local/vr_box_eval'))
    sys.path.insert(0,str(IGRIS/'igris_curobo/src'))
    from render_results import model_for
    out=OUTPUT/'native'/name;meta=json.loads((out/'metadata.json').read_text());metrics=json.loads((out/'metrics.json').read_text())
    inputs=json.loads((out/'input.json').read_text());it=np.array([r['t'] for r in inputs])
    with np.load(out/'physics.npz') as f:t=f['t'];qs=f['qpos'];forces=f['forces']
    with np.load(OUTPUT/'box_plans'/plan/'ik.npz') as f:pt=f['time'];goals=f['goals']
    model=model_for(out);data=mujoco.MjData(model);model.vis.global_.offwidth=960;model.vis.global_.offheight=720
    option=mujoco.MjvOption();option.geomgroup[2:]=0
    model.geom_rgba[model.geom_group==1,3]=1.
    renderer=mujoco.Renderer(model,720,960);cam=mujoco.MjvCamera();cam.lookat[:]=[.1,0,.87];cam.distance=2.6;cam.azimuth=140;cam.elevation=-15
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    dst=out/'measured_box.mp4';ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r',str(fps),'-i','-','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',str(dst)],stdin=subprocess.PIPE)
    saved_image=False
    for sec in np.arange(t[0],t[-1],speed/fps):
        i=min(np.searchsorted(t,sec),len(t)-1);k=min(np.searchsorted(it,sec),len(it)-1);frame=inputs[k]
        motion_time=frame.get('source_motion_time_s',frame['t']);g=goals[min(np.searchsorted(pt,motion_time),len(pt)-1)]
        data.qpos[:]=qs[i];mujoco.mj_forward(model,data);renderer.update_scene(data,camera=cam,scene_option=option)
        torso=model.body('torso').id;rotation=data.xmat[torso].reshape(3,3);errors=[]
        for side,ee in enumerate(EE_NAMES):
            actual=data.xpos[model.body(ee).id];goal=data.xpos[torso]+rotation@g[side,:3];errors.append(np.linalg.norm(actual-goal)*1000)
            geom=renderer.scene.geoms[renderer.scene.ngeom];renderer.scene.ngeom+=1
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.012),goal,np.eye(3).ravel(),np.array([.1,.9,.9,1] if side==0 else [1,.1,.8,1],np.float32))
        im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,116),fill=(18,24,31))
        draw.text((14,8),f'{label} | 1 kg | measured native MuJoCo 3.2.7 | playback {speed:g}x',font=font,fill='white')
        draw.text((14,36),f'Motion {motion_time:5.1f}s | {frame["stage"]} | EE L/R {errors[0]:.1f}/{errors[1]:.1f} mm',font=font,fill='white')
        draw.text((14,64),f'Box contact L/R {forces[i,0]:.1f}/{forces[i,1]:.1f} N | support {forces[i,6]:.1f} N',font=font,fill='white')
        draw.text((14,90),f'Trial result: {"PASS" if metrics["task_success"] else "FAIL"} | cyan/magenta: requested torso EE positions',font=font,fill=(120,255,160) if metrics['task_success'] else (255,150,130))
        ff.stdin.write(im.tobytes())
        if i>=len(t)//2 and not saved_image:im.save(out/'measured_box.png');saved_image=True
    ff.stdin.close();assert ff.wait()==0;renderer.close();print(dst)

def retarget_preview():
    import mujoco
    from PIL import Image,ImageDraw,ImageFont
    from .train import result_rows
    rows=result_rows();candidates=sorted([r for r in rows.values() if r['status']=='processed' and r['paper'] and r['frames']>=180],key=lambda r:r['path'])
    accepted=next(r for r in candidates if r['valid_frames']/r['frames']>.8)
    rejected=next(r for r in candidates if r['rejection_counts']['self_collision']>30)
    model=mujoco.MjModel.from_xml_path(str(OUTPUT/'model/igris_c_rcv_hand_torso.xml'));data=mujoco.MjData(model)
    model.vis.global_.offwidth=960;model.vis.global_.offheight=720;model.vis.headlight.ambient[:]=.6
    model.geom_rgba[model.geom_group==1,:3]=.65;model.geom_rgba[model.geom_group==1,3]=1.
    option=mujoco.MjvOption();option.geomgroup[:]=0;option.geomgroup[1]=1
    renderer=mujoco.Renderer(model,720,960);cam=mujoco.MjvCamera();cam.distance=2.7;cam.azimuth=170;cam.elevation=-15
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    dst=OUTPUT/'figures/retarget_preview.mp4';dst.parent.mkdir(exist_ok=True)
    ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','30','-i','-','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',str(dst)],stdin=subprocess.PIPE)
    manifest=[]
    for r in (accepted,rejected):
        f=np.load(r['output']);start=0 if r is accepted else max(0,int(np.flatnonzero(f['reject_flags'][:,3])[0])-30)
        indices=range(start,min(start+240,len(f['qpos'])));manifest.append(dict(path=r['path'],start=start,frames=len(indices),purpose='illustrative raw retarget; full-data audit is separate'))
        for i in indices:
            data.qpos[:]=f['qpos'][i];mujoco.mj_kinematics(model,data);cam.lookat[:]=data.xpos[model.body('torso').id]+[0,0,-.2]
            renderer.update_scene(data,camera=cam,scene_option=option)
            flags=f['reason_names'][f['reject_flags'][i]].tolist()
            if f['target_angle_error'][i,:,0].max()>np.deg2rad(40):flags.append('elbow_orientation')
            im=Image.fromarray(renderer.render());d=ImageDraw.Draw(im);d.rectangle((0,0,960,90),fill=(18,24,31))
            d.text((12,8),'Raw GMR retarget | '+r['path'],font=font,fill='white')
            d.text((12,36),f'Frame {i} | '+('REJECT: '+', '.join(flags) if flags else 'PASS all configured frame gates'),font=font,fill=(255,130,120) if flags else (130,240,160))
            d.text((12,63),'Rejected frames are shown for inspection and never bridge a training window.',font=font,fill='white');ff.stdin.write(im.tobytes())
    ff.stdin.close();assert ff.wait()==0;renderer.close();save_json(OUTPUT/'figures/retarget_preview.json',manifest);print(dst)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['plots','video','retarget']);ap.add_argument('--trial');ap.add_argument('--plan');ap.add_argument('--label')
    ap.add_argument('--speed',type=float,default=3.);ap.add_argument('--fps',type=int,default=15);a=ap.parse_args()
    if a.command=='plots':plots()
    elif a.command=='retarget':retarget_preview()
    else:video(a.trial,a.plan,a.label,speed=a.speed,fps=a.fps)
