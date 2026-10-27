"""Independent small SPD system: Krylov iterates versus a dense direct solve."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
from experiment import solve, dot, save_json

class DenseProblem:
    def __init__(self):
        a=torch.tensor([[1.,.2,0.],[0.,.3,1.],[.4,0.,.2],[0.,.1,.5]],device='cuda')
        self.matrix=a.T@a+.1*torch.eye(3,device='cuda')
        self.rhs=torch.tensor([[.1,.3,-.1],[.5,-.2,.3],[.1,.2,.1]],device='cuda')
        self.ridge=.1
        self.majorizer=self.matrix.sum(1,keepdim=True).expand_as(self.rhs)
        self.L=self.majorizer.max().item()
    def H(self,x): return self.matrix@x
    def objective(self,x): return (.5*dot(x,self.H(x))-dot(self.rhs,x)).item()

q=DenseProblem(); target=torch.linalg.solve(q.matrix,q.rhs)
errors={}
for method in ['GD','Nesterov','Adam','CG','PCG']:
    steps=3 if method in ['CG','PCG'] else 500
    x,hist=solve(q,torch.zeros_like(q.rhs),method,steps,.03,log_every=1)
    errors[method]=torch.linalg.norm(x-target).item()
    assert errors[method]<(2e-5 if method in ['CG','PCG'] else .002),(method,errors)
save_json(Path(__file__).parents[1]/'results/dense_solver_checks.json',errors)
print(errors)
