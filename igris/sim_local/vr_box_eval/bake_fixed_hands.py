"""Bake the shared fixed hand posture into the vendor's flat hand mesh bodies."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from igris_ik.assets import load_visual_urdf, FIXED_HAND_JOINT_VALUES, HAND_TOOL_FRAMES


def sync_hand_frames(path):
    """Copy fixed EE/camera frames into the vendor model without changing its axes."""
    import re
    urdf=load_visual_urdf();urdf.update_cfg(FIXED_HAND_JOINT_VALUES)
    source=path.read_text()
    names=[*HAND_TOOL_FRAMES, *(n for n in urdf.link_map if n.endswith('_cam'))]
    for name in names:
        side='l' if name==HAND_TOOL_FRAMES[0] or name.startswith('l_') else 'r'
        link=urdf.link_map[name];matrix=urdf.get_transform(name,side+'_wrist_connector')
        body=ET.Element('body',name=name,pos=' '.join(map(str,matrix[:3,3])),
                        quat=' '.join(map(str,Rotation.from_matrix(matrix[:3,:3]).as_quat()[[3,0,1,2]])))
        if link.inertial is not None:
            inertial=link.inertial
            ET.SubElement(body,'inertial',mass=str(inertial.mass),pos=' '.join(map(str,inertial.origin[:3,3])),
                          fullinertia=' '.join(map(str,inertial.inertia[[0,1,2,0,0,1],[0,1,2,1,2,2]])))
        for visual in link.visuals:
            geometry=visual.geometry
            attributes=dict(name=name+'_visual',contype='0',conaffinity='0',group='1',
                            pos=' '.join(map(str,visual.origin[:3,3])),
                            quat=' '.join(map(str,Rotation.from_matrix(visual.origin[:3,:3]).as_quat()[[3,0,1,2]])))
            if geometry.mesh is not None:
                mesh=geometry.mesh
                filename=mesh.filename.replace('package://igris_c_description/', '/igris/igris_curobo/src/igris_ik/data/igris_c_description/')
                if f'name="{name}_mesh"' not in source:
                    asset=ET.Element('mesh',name=name+'_mesh',file=filename,scale=' '.join(map(str,mesh.scale)))
                    source=source.replace('</asset>',ET.tostring(asset,encoding='unicode')+'\n  </asset>',1)
                attributes.update(type='mesh',mesh=name+'_mesh',rgba='.25 .25 .28 1')
            else:
                attributes.update(type='sphere',size=str(geometry.sphere.radius),rgba='1 0 0 1')
            ET.SubElement(body,'geom',attributes)
        source=re.sub(r'<body name="'+name+r'"[^>]*>.*?</body>\s*','',source,flags=re.S)
        source=source.replace(f'<body name="{side}_hand_rcv"',ET.tostring(body,encoding='unicode')+'\n<body name="'+side+'_hand_rcv"',1)
    path.write_text(source)


def bake(path):
    u=load_visual_urdf();u.update_cfg(FIXED_HAND_JOINT_VALUES)
    source=path.read_text();root=ET.fromstring(source)
    assets={Path(a.get('file')).name:a.get('name') for a in root.findall('./asset/mesh') if 'rcvh_' in a.get('name','')}
    for side in ('l','r'):
        body=root.find(f'.//body[@name="{side}_hand_rcv"]')
        template=dict(body.find('geom').attrib)
        for g in list(body.findall('geom')):body.remove(g)
        for link in u.robot.links:
            if not link.name.startswith(side+'_hand_') or link.name.endswith('_cam'):continue
            transform=u.get_transform(link.name,side+'_hand_root')
            for visual in link.visuals:
                mesh=visual.geometry.mesh
                if mesh is None:continue
                matrix=transform@visual.origin
                geom={**template,'name':f'{side}_hand_rcv_g{len(body.findall("geom"))}',
                      'mesh':assets[Path(mesh.filename).name],
                      'pos':' '.join(map(str,matrix[:3,3])),
                      'quat':' '.join(map(str,Rotation.from_matrix(matrix[:3,:3]).as_quat()[[3,0,1,2]]))}
                ET.SubElement(body,'geom',geom)
        # Preserve the vendor's 0.6 kg hand mass; update COM/inertia for the fixed pose.
        parts=[]
        for link in u.robot.links:
            if not link.name.startswith(side+'_hand_') or link.name.endswith('_cam') or link.inertial is None:continue
            x=u.get_transform(link.name,side+'_hand_root')@link.inertial.origin
            parts.append((link.inertial.mass,x[:3,3],x[:3,:3]@link.inertial.inertia@x[:3,:3].T))
        mass=sum(p[0] for p in parts);center=sum(w*p for w,p,_ in parts)/mass
        inertia=sum(I+w*(np.dot(p-center,p-center)*np.eye(3)-np.outer(p-center,p-center)) for w,p,I in parts)*(.6/mass)
        inertial=body.find('inertial');inertial.attrib.clear()
        inertial.attrib.update(mass='.6',pos=' '.join(map(str,center)),fullinertia=' '.join(map(str,inertia[[0,1,2,0,0,1],[0,1,2,1,2,2]])))
    import re
    for side in ('l','r'):
        body=root.find(f'.//body[@name="{side}_hand_rcv"]')
        ET.indent(body,space='  ')
        block=ET.tostring(body,encoding='unicode').rstrip()
        source=re.sub(r'<body name="'+side+r'_hand_rcv".*?</body>',lambda match:block,source,flags=re.S)
    path.write_text(source)
    sync_hand_frames(path)


def bake_collision_spheres():
    """Refit the existing hand-root collision approximation to the same meshes."""
    import re,yaml,mujoco
    from scipy.cluster.vq import kmeans2
    from igris_ik.sim.mujoco_model import load_fixed_hand_model
    from igris_ik.model import model_yaml
    model=load_fixed_hand_model().model;data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    path=model_yaml();source=path.read_text()
    for side in ('l','r'):
        root=model.body(side+'_hand_root').id;rotation=data.xmat[root].reshape(3,3);points=[]
        for g in range(model.ngeom):
            name=model.body(model.geom_bodyid[g]).name
            if not name.startswith(side+'_hand_') or name.endswith('_cam') or model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            mesh=model.geom_dataid[g];v=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
            points.append((v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]-data.xpos[root])@rotation)
        points=np.unique(np.concatenate(points).round(7),axis=0)
        centers,labels=kmeans2(points,12,iter=30,minit='++',seed=0)
        spheres=[dict(center=c.tolist(),radius=float(np.linalg.norm(points[labels==i]-c,axis=1).max()+.002)) for i,c in enumerate(centers)]
        replacement='\n'.join('    '+line for line in yaml.safe_dump({side+'_hand_root':spheres},sort_keys=False).splitlines())+'\n'
        source=re.sub(r'^    '+side+r'_hand_root:\n(?:    -.*\n|      .*\n)+',replacement,source,count=1,flags=re.M)
    path.write_text(source)


if __name__=='__main__':
    bake(Path(__file__).resolve().parents[1]/'igris_c_v2_rcv_hand.xml')
    bake_collision_spheres()
