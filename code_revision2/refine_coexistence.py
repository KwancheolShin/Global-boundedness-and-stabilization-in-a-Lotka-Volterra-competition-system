from pathlib import Path
import sys,json
import numpy as np
from scipy.optimize import root as solve_root
from scipy.linalg import eigvals
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
import sim
base=json.loads((root/'results'/'long1d_diagnostics.json').read_text())
out={}
for N in [200,400]:
    for case in ['M2','M3']:
        p=sim.base_B(False); p['d1'],p['d2']=sim.MOT[case]
        x,rhs=sim.make_rhs(N,p)
        coarse=np.array(base[case+'_N100']['final_state']).reshape(3,100)
        xc=(np.arange(100)+.5)/100
        z0=np.r_[tuple(np.interp(x,xc,f) for f in coarse)]
        result=solve_root(lambda z:rhs(0,z),z0,tol=1e-10)
        z=result.x
        rec={'N':N,'root_success':bool(result.success),'residual_inf':float(abs(rhs(0,z)).max()),
             'masses':[float(z[:N].mean()),float(z[N:2*N].mean())],
             'min_final':[float(z[i*N:(i+1)*N].min()) for i in range(3)]}
        out[f'{case}_N{N}']=rec
        print(case,N,rec,flush=True)
        (root/'results'/'coexistence_refinement.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
