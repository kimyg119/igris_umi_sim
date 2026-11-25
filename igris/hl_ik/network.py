"""FiSTA reconstruction from HL-IK v3 text and Figure 4.

The paper omits widths, heads, dropout probability and optimization settings.
Those are explicit IGRIS choices, not recovered author hyperparameters.
"""
from dataclasses import dataclass,asdict
import torch
from torch import nn

@dataclass
class NetworkConfig:
    history: int=5
    hidden: int=128
    heads: int=4
    dropout: float=.1
    arm_mode: str='independent_single_arm_text_spec'

class FiSTA(nn.Module):
    def __init__(self,config):
        super().__init__();h=config.hidden
        self.temporal=nn.GRU(14,h,batch_first=True)
        self.spatial_pre=nn.Sequential(nn.Linear(7,h),nn.ReLU(),nn.Linear(h,h))
        self.attention=nn.MultiheadAttention(h,config.heads,batch_first=True)
        self.spatial_post=nn.Sequential(nn.Linear(2*h,h),nn.ReLU(),nn.Linear(h,h))
        self.goal=nn.Sequential(nn.Linear(7,h),nn.ReLU(),nn.Linear(h,2*h))
        self.output=nn.Sequential(nn.Linear(2*h,h),nn.ReLU(),nn.Dropout(config.dropout),nn.Linear(h,7))

    def forward(self,history,goal):
        _,hidden=self.temporal(history)
        tokens=self.spatial_pre(history[:,-1].reshape(-1,2,7))
        spatial,_=self.attention(tokens,tokens,tokens,need_weights=False)
        spatial=self.spatial_post(spatial.flatten(1))
        scale,shift=self.goal(goal).chunk(2,dim=-1)
        conditioned=scale.sigmoid()*hidden[-1]+shift
        return self.output(torch.cat((conditioned,spatial),dim=-1))

class BimanualFiSTA(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or NetworkConfig()
        self.arms=nn.ModuleList([FiSTA(self.config),FiSTA(self.config)])
    def forward(self,history,goal):
        return torch.stack([arm(history[:,:,s],goal[:,s]) for s,arm in enumerate(self.arms)],dim=1)

def canonicalize(history,goal,label=None):
    """Use latest-history quaternion hemisphere; no current elbow in inputs."""
    h=history.clone();g=goal.clone();y=None if label is None else label.clone()
    for start in (3,10):
        latest=h[:,-1,:,start:start+4].clone()
        latest*=torch.where(latest[:,:,:1]<0,-1.,1.)
        sign=torch.where((h[:,:,:,start:start+4]*latest[:,None]).sum(-1,keepdim=True)<0,-1.,1.)
        h[:,:,:,start:start+4]*=sign
    sign=torch.where((g[:,:,3:]*h[:,-1,:,3:7]).sum(-1,keepdim=True)<0,-1.,1.)
    g[:,:,3:]*=sign
    if y is not None:
        sign=torch.where((y[:,:,3:]*h[:,-1,:,10:14]).sum(-1,keepdim=True)<0,-1.,1.)
        y[:,:,3:]*=sign
    return h,g,y

def batch_from_states(states,starts,mean,std):
    # t is the prediction time; history is exactly t-5 through t-1.
    history=states[starts[:,None]+torch.arange(5,device=states.device)[None]]
    future=states[starts+5];goal=future[:,:,:7];label=future[:,:,7:]
    history,goal,label=canonicalize(history,goal,label)
    return (history-mean)/std,(goal-mean[:,:7])/std[:,:7],(label-mean[:,7:])/std[:,7:],label

def decode(normalized,mean,std):
    pose=normalized*std[:,7:]+mean[:,7:]
    norm=pose[:,:,3:].norm(dim=-1,keepdim=True)
    if torch.any(norm<1e-8):raise ValueError('near-zero predicted quaternion')
    return torch.cat((pose[:,:,:3],pose[:,:,3:]/norm),dim=-1)
