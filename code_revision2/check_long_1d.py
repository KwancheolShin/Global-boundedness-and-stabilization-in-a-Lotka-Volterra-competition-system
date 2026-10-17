"""Check the unresolved tails of heterogeneous local competition."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.linalg import eigvals
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
import sim
out={}
T=1e7
te=np.r_[0,np.geomspace(.1,T,500)[:-1],T]
for N in [100,200]:
    for case in ['M2','M3']:
        p=sim.base_B(False)
        p['d1'],p['d2']=sim.MOT[case]
        rtol,atol=(1e-10,1e-13) if N==100 else (1e-8,1e-11)
        x,sol=sim.run(N,p,T,te,rtol=rtol,atol=atol)
        _,rhs=sim.make_rhs(N,p)
        y=sol.y[:,-1]
        J=np.empty((y.size,y.size))
        for i in range(y.size):
            z=y.astype(complex); z[i]+=1e-30j
            J[:,i]=np.imag(rhs(0,z))/1e-30
        ev=eigvals(J)
        rec={'N':N,'T':T,'rtol':rtol,'atol':atol,'final_masses':[float(y[:N].mean()),float(y[N:2*N].mean())],
             'min_final':[float(y[i*N:(i+1)*N].min()) for i in range(3)],
             'rhs_residual_inf':float(abs(rhs(T,y)).max()),
             'leading_eigenvalues':[[float(z.real),float(z.imag)] for z in sorted(ev,key=lambda z:z.real,reverse=True)[:5]],
             't':te.tolist(),'mass_u':sol.y[:N].mean(axis=0).tolist(),'mass_v':sol.y[N:2*N].mean(axis=0).tolist(),
             'final_state':y.tolist()}
        out[f'{case}_N{N}']=rec
        (root/'results'/'long1d_diagnostics.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
        print(case,N,{k:v for k,v in rec.items() if k not in ['t','mass_u','mass_v','final_state']},flush=True)
