"""Long-time mesh refinement and exploratory invasion eigenvalues."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import root as find_root
from scipy.linalg import eigvalsh

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root))
import sim
import sim2d
import sim2d_bdf2 as b

out = {'invasion_diagnostics':{},'long_time_mesh':{}}
def save():
    (root/'results'/'extra_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
for N in (100,200):
    h=1/N
    x=(np.arange(N)+.5)*h
    L=(np.diag(-2*np.ones(N))+np.diag(np.ones(N-1),1)+np.diag(np.ones(N-1),-1))/h**2
    L[0,0]=L[-1,-1]=-1/h**2
    res=1+.5*np.cos(np.pi*x)
    for case in ('M1','M2','M3'):
        du,dv=sim.MOT[case]
        for resident,dr,di in [('u',du,dv),('v',dv,du)]:
            def f(z):
                R,W=z[:N],z[N:]
                return np.r_[L@(dr(W)*R)+R*(W-R),L@W+W*(res-R-W)]
            sol=find_root(f,np.r_[np.full(N,.5),np.full(N,.5)],tol=1e-10)
            R,W=sol.x[:N],sol.x[N:]
            D=di(W)
            sd=np.sqrt(D)
            symmetric=sd[:,None]*L*sd[None,:]+np.diag(W-R)
            inv=float(eigvalsh(symmetric,subset_by_index=[N-1,N-1])[0])
            rec={'N':N,'resident':resident,'root_success':bool(sol.success),
                 'residual_inf':float(np.max(abs(f(sol.x)))),
                 'minimum_R':float(R.min()),'minimum_W':float(W.min()),
                 'resident_mass':float(R.mean()),'absent_invasion_eigenvalue':inv}
            key=f'{case}_resident_{resident}_N{N}'
            out['invasion_diagnostics'][key]=rec
            print(key,rec,flush=True)
            save()

for nl in (False,True):
    for case in ('M1','M2'):
        mode='nonlocal' if nl else 'local'
        p=sim2d.base_B(nl)
        p['d1']=lambda w:.1+0*w
        p['dd1']=lambda w:0*w
        if case=='M1':
            p['d2']=lambda w:1+0*w
            p['dd2']=lambda w:0*w
        else:
            p['d2']=lambda w:np.exp(-2*w)
            p['dd2']=lambda w:-2*np.exp(-2*w)
        N=48; M=N*N; T=1e7
        xx,yy,sol=b.run(N,p,T,[T],1e-6,1e-12)
        masses=[float(sol.y[:M,-1].mean()),float(sol.y[M:2*M,-1].mean())]
        ref=json.loads((root/'results'/f'long2d_{mode}_{case}.json').read_text())
        old=[ref['mass_u'][-1],ref['mass_v'][-1]]
        rec={'T':T,'N':N,'mass_N48':masses,'mass_N32':old,
             'max_difference':float(np.max(np.abs(np.array(masses)-old))),
             'final_mass_drift':[float(sol.y[-2,-1]-masses[0]),float(sol.y[-1,-1]-masses[1])]}
        out['long_time_mesh'][mode+'_'+case]=rec
        print(mode,case,rec,flush=True)
        save()
