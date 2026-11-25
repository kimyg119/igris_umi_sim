"""Render the built torso model, fixed hand posture and shared EE axes."""
import numpy as np
import mujoco
import imageio.v3 as iio
from .common import *

def main():
    model=mujoco.MjModel.from_xml_path(str(OUTPUT/'model/igris_c_rcv_hand_torso.xml'))
    data=mujoco.MjData(model);data.qpos[2]=.85
    for name,value in [('l_shoulder_pitch',-.5),('r_shoulder_pitch',-.5),('l_shoulder_roll',.25),('r_shoulder_roll',-.25),('l_elbow_pitch',-1.3),('r_elbow_pitch',-1.3)]:
        data.qpos[model.joint(name).qposadr[0]]=value
    mujoco.mj_forward(model,data)
    model.vis.global_.offwidth=1200;model.vis.global_.offheight=1000
    camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.77];camera.distance=2.15;camera.azimuth=180;camera.elevation=-12
    option=mujoco.MjvOption();option.geomgroup[0]=0;option.geomgroup[1]=1;option.geomgroup[2:]=0
    model.vis.headlight.active=1
    model.vis.headlight.ambient[:]=.6;model.vis.headlight.diffuse[:]=.8
    for i in range(model.ngeom):
        if model.geom_group[i]==1:
            model.geom_rgba[i,:3]=np.maximum(model.geom_rgba[i,:3],.45)
            model.geom_rgba[i,3]=1.
    renderer=mujoco.Renderer(model,height=1000,width=1200);renderer.update_scene(data,camera,option)
    for name in EE_NAMES:
        b=model.body(name).id;p=data.xpos[b];r=data.xmat[b].reshape(3,3)
        for axis,color in enumerate(([1,.12,.12,1],[.1,.9,.2,1],[.15,.4,1,1])):
            geom=renderer.scene.geoms[renderer.scene.ngeom];renderer.scene.ngeom+=1
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.ones(3),np.zeros(3),np.eye(3).flatten(),np.array(color,dtype=np.float32))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.004,p,p+.10*r[:,axis])
    iio.imwrite(OUTPUT/'model/torso_rcv_hand_axes.png',renderer.render());renderer.close()
    print(OUTPUT/'model/torso_rcv_hand_axes.png')

if __name__=='__main__':main()
