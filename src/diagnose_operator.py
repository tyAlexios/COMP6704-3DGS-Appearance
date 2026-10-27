import sys,math,json
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).parent))
from experiment import ROOT,Scene,GaussianModel,FrozenRenderer,Quadratic,dot,evenly
scene_name=sys.argv[1]
args=SimpleNamespace(source_path='/home/tianyi/data/3D_Research/DATASETS/tandt/'+scene_name,
    model_path=str(ROOT/('geometry_'+scene_name)),images='images',depths='',eval=True,
    train_test_exp=False,white_background=False,resolution=256,data_device='cuda')
gs=GaussianModel(0,'default'); s=Scene(args,gs,load_iteration=3000,shuffle=False)
c0=(gs.get_features[:,0,:]*.28209479177387814+.5).detach()
cams=sorted(s.getTrainCameras(),key=lambda c:c.image_name)
val=cams[::11]; cams=evenly([c for c in cams if c not in val],8)
R=FrozenRenderer(gs,cams,c0); q=Quadratic(R,cams,.01)
torch.manual_seed(918)
for positive in [False,True]:
    u=(torch.rand_like(c0) if positive else torch.randn_like(c0))*.1
    v=(torch.rand_like(c0) if positive else torch.randn_like(c0))*.1
    cam=cams[0]; au=R.forward(u,cam)
    t=torch.rand_like(au) if positive else torch.randn_like(au)
    p1=dot(au,t).item(); p2=dot(u,R.adjoint(t,cam)).item()
    hu=q.H(u); analytic=(dot(hu-q.rhs,v)/q.nobs).item()
    print('positive',positive,'adjoint',p1,p2,'norms',torch.linalg.norm(au).item(),torch.linalg.norm(t).item(),'analytic',analytic,flush=True)
    for eps in [.001,.01,.1,1.]:
        plus=q.objective(u+eps*v); minus=q.objective(u-eps*v)
        print('FD',eps,plus,minus,(plus-minus)/(2*eps),flush=True)
