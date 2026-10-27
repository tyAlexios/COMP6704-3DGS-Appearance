"""Fixed-geometry RGB optimization with the installed 3DGS CUDA rasterizer.

No source data or previous experiment is written. All color iterates are
unconstrained; clamping is reserved for a separately labelled display metric.
Python 3.7 / PyTorch 1.13 compatible. See README for the exact objective.
"""
import argparse, csv, hashlib, json, math, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT / 'reference_deps'))
from scene import Scene, GaussianModel
import diff_gaussian_rasterization
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.loss_utils import ssim


def dot(x, y):
    return (x.double() * y.double()).sum()


def sync():
    torch.cuda.synchronize()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


class FrozenRenderer:
    """Linear RGB forward operator and its cached exact custom-backward adjoint."""
    def __init__(self, gs, cameras, c0):
        self.gs, self.cameras, self.c0 = gs, cameras, c0
        self.rasters = {}
        self.graphs = {}
        for cam in cameras:
            settings_args = dict(
                image_height=int(cam.image_height), image_width=int(cam.image_width),
                tanfovx=math.tan(cam.FoVx / 2), tanfovy=math.tan(cam.FoVy / 2),
                bg=torch.zeros(3, device='cuda'), scale_modifier=1.,
                viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
                sh_degree=0, campos=cam.camera_center, prefiltered=False,
                debug=False)
            if 'antialiasing' in GaussianRasterizationSettings._fields:
                settings_args['antialiasing']=False
            settings = GaussianRasterizationSettings(**settings_args)
            self.rasters[cam.image_name] = GaussianRasterizer(settings)
        self.xyz = gs.get_xyz.detach()
        self.opacity = gs.get_opacity.detach()
        self.scales = gs.get_scaling.detach()
        self.rotations = gs.get_rotation.detach()
        self.screen = torch.zeros_like(self.xyz)

    def raw(self, color, cam):
        out = self.rasters[cam.image_name](
            means3D=self.xyz, means2D=self.screen, shs=None,
            colors_precomp=color.contiguous(), opacities=self.opacity,
            scales=self.scales, rotations=self.rotations, cov3D_precomp=None)
        return out[0]

    def forward(self, x, cam):
        with torch.no_grad():
            return self.raw(x, cam)

    def adjoint(self, image, cam):
        name = cam.image_name
        if name not in self.graphs:
            probe = torch.zeros_like(self.c0, requires_grad=True)
            self.graphs[name] = (probe, self.raw(probe, cam))
        probe, rendered = self.graphs[name]
        return torch.autograd.grad(rendered, probe, image.contiguous(), retain_graph=True)[0].detach()


class Quadratic:
    def __init__(self, renderer, cams, ridge):
        self.R, self.cams, self.ridge = renderer, cams, ridge
        self.nobs = sum(c.original_image.numel() for c in cams)
        self.targets = [c.original_image.cuda() - renderer.forward(renderer.c0, c) for c in cams]
        self.rhs = torch.zeros_like(renderer.c0)
        for c, b in zip(cams, self.targets):
            self.rhs += renderer.adjoint(b, c)
        # Nonnegative A implies diag(H 1) - H is PSD (a graph Laplacian).
        self.majorizer = self.H(torch.ones_like(renderer.c0)).clamp_min(ridge)
        self.L = self.majorizer.max().item()

    def H(self, x):
        v = self.ridge * x
        for c in self.cams:
            v = v + self.R.adjoint(self.R.forward(x, c), c)
        return v

    def objective(self, x):
        total = self.ridge * dot(x, x)
        for c, b in zip(self.cams, self.targets):
            err = self.R.forward(x, c) - b
            total += dot(err, err)
        return (0.5 * total / self.nobs).item()


def validate_operator(q, out):
    torch.manual_seed(918)
    u = torch.randn_like(q.rhs) * .1
    v = torch.randn_like(q.rhs) * .1
    cam = q.cams[0]
    a, b = .37, -.23
    lhs = q.R.forward(a*u+b*v, cam)
    rhs = a*q.R.forward(u, cam) + b*q.R.forward(v, cam)
    lin = (torch.linalg.norm(lhs-rhs)/torch.linalg.norm(rhs)).item()
    image = torch.randn_like(lhs)
    ip1 = dot(q.R.forward(u, cam), image).item()
    ip2 = dot(u, q.R.adjoint(image, cam)).item()
    adj = abs(ip1-ip2)/max(abs(ip1),abs(ip2),1e-12)
    hu, hv = q.H(u), q.H(v)
    sym = abs((dot(u,hv)-dot(hu,v)).item()) / max(abs(dot(u,hv).item()),1e-12)
    eps=.01
    fd=(q.objective(u+eps*v)-q.objective(u-eps*v))/(2*eps)
    an=(dot(hu-q.rhs,v)/q.nobs).item()
    grad=abs(fd-an)/max(abs(fd),abs(an),1e-12)
    checks={'linearity_relative_error':lin,'adjoint_relative_error':adj,
            'symmetry_relative_error':sym,'gradient_relative_error':grad,
            'positive_curvature':dot(u,hu).item(), 'L_upper':q.L,
            'majorizer_min':q.majorizer.min().item()}
    save_json(out, checks)
    assert lin<1e-4 and adj<5e-4 and sym<5e-4 and grad<5e-4, checks
    return checks


def solve(q, x0, method, steps, lr=0.03, log_every=10, tol=0):
    x=x0.clone(); prev=x.clone(); m=torch.zeros_like(x); v=m.clone()
    elapsed=0.; history=[]
    sync(); start=time.perf_counter()
    g=q.H(x)-q.rhs
    sync(); elapsed += time.perf_counter()-start
    g0=math.sqrt(dot(g,g).item())
    r=-g; z=r/q.majorizer if method=='PCG' else r.clone(); p=z.clone(); rz=dot(r,z)
    beta_n=(math.sqrt(q.L)-math.sqrt(q.ridge))/(math.sqrt(q.L)+math.sqrt(q.ridge))
    for k in range(steps+1):
        if k % log_every == 0 or k==steps:
            trueg=q.H(x)-q.rhs
            rel=math.sqrt(dot(trueg,trueg).item())/max(g0,1e-30)
            history.append({'iteration':k,'solver_seconds':elapsed,
                            'objective':q.objective(x),'relative_gradient':rel})
            if not math.isfinite(history[-1]['objective']):
                raise RuntimeError('Non-finite objective')
            if tol and rel<tol: break
        if k==steps: break
        sync(); start=time.perf_counter()
        with torch.no_grad():
            if method in ['CG','PCG']:
                hp=q.H(p)
                denom=dot(p,hp)
                if denom.item()<=0 or rz.item()<1e-30: break
                alpha=(rz/denom).float()
                x=x+alpha*p; r=r-alpha*hp
                if (k+1)%25==0:
                    # Restart after true residual replacement to limit float32 drift.
                    r=q.rhs-q.H(x)
                    z=r/q.majorizer if method=='PCG' else r.clone()
                    p=z.clone(); rz=dot(r,z)
                else:
                    z=r/q.majorizer if method=='PCG' else r.clone()
                    rznew=dot(r,z)
                    p=z+(rznew/rz).float()*p; rz=rznew
            elif method=='GD':
                x=x-(q.H(x)-q.rhs)/q.L
            elif method=='Nesterov':
                y=x+beta_n*(x-prev)
                nxt=y-(q.H(y)-q.rhs)/q.L
                prev=x; x=nxt
            elif method=='Adam':
                g=q.H(x)-q.rhs
                m=.9*m+.1*g; v=.999*v+.001*g*g
                mh=m/(1-.9**(k+1)); vh=v/(1-.999**(k+1))
                x=x-lr*mh/(torch.sqrt(vh)+1e-8)
            else: raise ValueError(method)
        sync(); elapsed += time.perf_counter()-start
    return x,history


def evaluate(R, delta, cams):
    vals=[]
    with torch.no_grad():
        for cam in cams:
            image=R.forward(R.c0+delta,cam)
            gt=cam.original_image.cuda()
            mse=((image-gt)**2).mean().item()
            clipped=image.clamp(0,1)
            cmse=((clipped-gt)**2).mean().item()
            vals.append({'image':cam.image_name,'raw_mse':mse,
                         'raw_psnr':-10*math.log10(max(mse,1e-20)),
                         'clipped_psnr':-10*math.log10(max(cmse,1e-20)),
                         'clipped_ssim':ssim(clipped,gt).item()})
    return {'raw_psnr':float(np.mean([v['raw_psnr'] for v in vals])),
            'clipped_psnr':float(np.mean([v['clipped_psnr'] for v in vals])),
            'clipped_ssim':float(np.mean([v['clipped_ssim'] for v in vals])), 'per_view':vals}


def evenly(cams, n):
    return [cams[i] for i in np.linspace(0,len(cams)-1,min(n,len(cams)),dtype=int)]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--scene',required=True)
    p.add_argument('--steps',type=int,default=100); p.add_argument('--width',type=int,default=256)
    p.add_argument('--views',type=int,default=8); p.add_argument('--ridge',type=float,default=.01)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]); p.add_argument('--tag',default='main')
    p.add_argument('--methods',nargs='+',default=['GD','Nesterov','Adam','CG','PCG'])
    p.add_argument('--adam-lr',type=float,default=.03); p.add_argument('--pilot',action='store_true')
    a=p.parse_args(); torch.set_num_threads(4)
    assert str(ROOT/'reference_deps') in diff_gaussian_rasterization.__file__, 'Reference extension not selected'
    out=ROOT/'results'/('{}__{}'.format(a.scene,a.tag)); out.mkdir(parents=True,exist_ok=False)
    ds=Path('/home/tianyi/data/3D_Research/DATASETS/tandt')/a.scene
    model=ROOT/('geometry_'+a.scene)
    args=SimpleNamespace(source_path=str(ds),model_path=str(model),images='images',depths='',
                         eval=True,train_test_exp=False,white_background=False,
                         resolution=a.width,data_device='cuda')
    gs=GaussianModel(0,'default'); scene=Scene(args,gs,load_iteration=3000,shuffle=False)
    for value in vars(gs).values():
        if isinstance(value,torch.Tensor): value.requires_grad_(False)
    trainall=sorted(scene.getTrainCameras(),key=lambda c:c.image_name)
    val=trainall[::11]; trainpool=[c for c in trainall if c not in val]
    train=evenly(trainpool,a.views); test=sorted(scene.getTestCameras(),key=lambda c:c.image_name)
    c0=(gs.get_features[:,0,:]*0.28209479177387814+.5).detach()
    R=FrozenRenderer(gs,train+val+test,c0)
    sync(); setup_start=time.perf_counter(); q=Quadratic(R,train,a.ridge)
    setup_seconds=time.perf_counter()-setup_start
    checks=validate_operator(q,out/'operator_checks.json')
    ckpt=model/'point_cloud/iteration_3000/point_cloud.ply'
    manifest={'arguments':vars(a),'torch':torch.__version__,'cuda':torch.version.cuda,
              'reference_rasterizer_commit':(ROOT/'reference_rasterizer_commit.txt').read_text().strip(),
              'rasterizer_module':diff_gaussian_rasterization.__file__,
              'python':sys.version,'gpu':torch.cuda.get_device_name(0),
              'gaussians':int(c0.shape[0]),'scalar_parameters':c0.numel(),'nobs':q.nobs,
              'geometry_checkpoint':str(ckpt),'geometry_sha256':sha(ckpt),
              'train_fit':[c.image_name for c in train], 'validation':[c.image_name for c in val],
              'test':[c.image_name for c in test], 'all_geometry_train':[c.image_name for c in trainall],
              'L_upper':q.L,'setup_seconds':setup_seconds,
              'majorizer_percentiles':np.percentile(q.majorizer.cpu().numpy(),[0,25,50,75,100]).tolist(),
              'input_hashes':{str(f.relative_to(ds)):sha(f) for f in sorted(ds.rglob('*')) if f.is_file() and f.suffix.lower() in ['.bin','.jpg','.png']}}
    save_json(out/'manifest.json',manifest)
    save_json(out/'baseline.json',{'test':evaluate(R,torch.zeros_like(c0),test),
                                  'validation':evaluate(R,torch.zeros_like(c0),val)})
    # Tune Adam on validation colors only; geometry already saw these views.
    adam_lr=a.adam_lr
    if a.pilot:
        torch.manual_seed(0); initial=torch.randn_like(c0)*.1
        pilots=[]
        for lr in [.003,.01,.03]:
            x,hist=solve(q,initial,'Adam',40,lr)
            score=evaluate(R,x,val)['raw_psnr']
            pilots.append({'lr':lr,'validation_psnr':score,'solver_seconds':hist[-1]['solver_seconds']})
        adam_lr=max(pilots,key=lambda x:x['validation_psnr'])['lr']
        save_json(out/'adam_pilot.json',{'trials':pilots,'selected_lr':adam_lr})
    summaries=[]
    for seed in a.seeds:
        torch.manual_seed(seed); initial=torch.randn_like(c0)*.1
        initial_test=evaluate(R,initial,test)
        offset=seed%len(a.methods)
        for method in a.methods[offset:]+a.methods[:offset]:
            print('START',a.scene,seed,method,flush=True)
            x,hist=solve(q,initial,method,a.steps,adam_lr)
            name='{}_seed{}'.format(method,seed)
            with open(str(out/(name+'.csv')),'w') as f:
                w=csv.DictWriter(f,fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
            torch.save(x.cpu(),str(out/(name+'.pt')))
            result={'scene':a.scene,'method':method,'seed':seed,'ridge':a.ridge,
                    'adam_lr':adam_lr,'initial_objective':hist[0]['objective'],
                    'initial_test':initial_test,
                    'final_objective':hist[-1]['objective'],'relative_gradient':hist[-1]['relative_gradient'],
                    'solver_seconds':hist[-1]['solver_seconds'],'test':evaluate(R,x,test),
                    'validation':evaluate(R,x,val),'history':hist}
            save_json(out/(name+'.json'),result); summaries.append(result)
            if seed==0:
                cam=test[len(test)//2]
                for label,im in [('prediction',R.forward(c0+x,cam)),('target',cam.original_image.cuda())]:
                    arr=(im.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                    Image.fromarray(arr).save(str(out/(name+'_'+label+'.png')))
            save_json(out/'summary.json',summaries)
            print('DONE',a.scene,seed,method,result['final_objective'],result['test']['raw_psnr'],result['solver_seconds'],flush=True)
    assert manifest['geometry_sha256']==sha(ckpt),'Geometry changed during color optimization'
    save_json(out/'complete.json',{'geometry_unchanged':True,'runs':len(summaries)})


if __name__=='__main__': main()
