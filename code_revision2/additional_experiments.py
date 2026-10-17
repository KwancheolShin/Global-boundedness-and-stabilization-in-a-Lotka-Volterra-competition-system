"""Revision: Holling II uptake and heterogeneous invasion diagnostics.

Usage: python additional_experiments.py holling|invasion|plots|all
Paths are resolved relative to this file. No network access is needed.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.optimize import root, brentq
from scipy.linalg import eigvalsh
import sim

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
FIG = HERE.parent / 'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2), encoding='utf-8')


def holling():
    p = dict(sim.A)
    h = 0.2
    F = lambda w: w / (1 + h*w)
    p['F1'] = p['F2'] = F
    ws = brentq(lambda w: 4*F(w)/2.9 - (2-w)*(1+h*w), 0, 2, xtol=1e-14)
    us = vs = 2*F(ws)/2.9
    te = np.linspace(0, 40, 401)
    out = {'h': h, 'steady': [us, vs, ws],
           'S1_lhs': us/4, 'S1_rhs': F(ws), 'meshes': {}}
    for N in (100, 200):
        x, sol = sim.run(N, p, 40, te, rtol=1e-10, atol=1e-13)
        errors = np.array([np.abs(sol.y[i*N:(i+1)*N]-z).max(axis=0)
                           for i,z in enumerate([us,vs,ws])])
        _, rhs = sim.make_rhs(N, p)
        rec = {'N': N, 'rtol': 1e-10, 'atol': 1e-13,
               'final_error_inf': float(errors[:,-1].max()),
               'minimum_over_run': float(sol.y.min()),
               'final_rhs_residual_inf': float(abs(rhs(40,sol.y[:,-1])).max()),
               'x': x.tolist(), 't': te.tolist(), 'errors': errors.tolist(),
               'profiles_t2': sol.y[:,20].reshape(3,N).tolist()}
        out['meshes'][str(N)] = rec
        print('Holling II', N, {k:rec[k] for k in ['final_error_inf','minimum_over_run','final_rhs_residual_inf']}, flush=True)
    out['mesh_difference_t2'] = float(np.max(abs(
        np.asarray(out['meshes']['100']['profiles_t2'])-
        np.asarray(out['meshes']['200']['profiles_t2']).reshape(3,100,2).mean(axis=2))))
    save('hollingII.json', out)
    print({k:v for k,v in out.items() if k!='meshes'}, flush=True)


def invasion():
    out = {}
    for N in (100,200):
        x = (np.arange(N)+0.5)/N
        r = 1+0.5*np.cos(np.pi*x)
        L = np.diag(-2*np.ones(N))+np.diag(np.ones(N-1),1)+np.diag(np.ones(N-1),-1)
        L[0,0] = L[-1,-1] = -1
        L *= N*N
        for case in ('M2','M3'):
            du,dv = sim.MOT[case]
            dr,di = (dv,du) if case=='M2' else (du,dv)
            def evaluate(c):
                def rhs(z):
                    R,W = z[:N],z[N:]
                    return np.r_[L@(dr(W)*R)+R*(W-R-c*R.mean()),
                                 L@W+W*(r-R-W)]
                z0 = np.r_[np.full(N,1/(2+c)),np.full(N,(1+c)/(2+c))]
                sol = root(rhs,z0,tol=1e-10)
                R,W = sol.x[:N],sol.x[N:]
                residual = float(abs(rhs(sol.x)).max())
                if min(R.min(),W.min()) <= 0 or residual > 2e-8:
                    raise RuntimeError((case,N,c,sol.message,residual))
                sd = np.sqrt(di(W))
                symmetric = sd[:,None]*L*sd[None,:]+np.diag(W-R-c*R.mean())
                ev = float(eigvalsh(symmetric,subset_by_index=[N-1,N-1])[0])
                return {'c': float(c), 'invasion_eigenvalue': ev,
                        'resident_mass': float(R.mean()), 'residual_inf': residual,
                        'minimum_resident': float(R.min()), 'minimum_resource':float(W.min()),
                        'root_success': bool(sol.success)}
            records = [evaluate(c) for c in np.linspace(0,0.5,26)]
            brackets = [(a['c'],b['c']) for a,b in zip(records,records[1:])
                        if a['invasion_eigenvalue']*b['invasion_eigenvalue']<0]
            crossings = []
            for lo,hi in brackets:
                cstar = brentq(lambda c:evaluate(c)['invasion_eigenvalue'],lo,hi,xtol=2e-7)
                crossings.append(evaluate(cstar))
            out[f'{case}_N{N}'] = {'N':N,'rare_species':'u' if case=='M2' else 'v',
                                  'records':records,'crossings':crossings}
            save('invasion_continuation.json',out)
            print(case,N,'lambda(0),lambda(.5):',records[0]['invasion_eigenvalue'],records[-1]['invasion_eigenvalue'],'crossings',crossings,flush=True)


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors = ['#185d93','#ba4d26','#418150']
    data = json.loads((OUT/'long1d_diagnostics.json').read_text())
    steady = json.loads((OUT/'coexistence_refinement.json').read_text())
    fig,axes = plt.subplots(1,2,figsize=(10,3.8),sharey=True,layout='constrained')
    for ax,case,sp,col in zip(axes,['M2','M3'],['u','v'],colors):
        for N,ls in [(100,'-'),(200,'--')]:
            d=data[f'{case}_N{N}']; t=np.asarray(d['t']); mask=t>0
            ax.loglog(t[mask],np.asarray(d['mass_'+sp])[mask],ls=ls,color=col if N==100 else 'black',lw=1.6,label=f'N = {N}')
        value=steady[f'{case}_N400']['masses'][0 if sp=='u' else 1]
        ax.axhline(value,color='gray',ls=':',label='Stationary solve, N = 400')
        ax.axvline(2e5,color='gray',ls='-.',lw=0.8)
        ax.set(xlim=(1,1e7),ylim=(.003,.7),xlabel='Time t',title=f'({"a" if case=="M2" else "b"}) Local {case}: rare species {sp}')
        ax.grid(alpha=.2); ax.legend(fontsize=8,loc='lower left')
    axes[0].set_ylabel('Total mass')
    for ext in ('pdf','png'): fig.savefig(FIG/f'rev2_fig6_coexistence.{ext}',dpi=180)
    plt.close(fig)
    continuation=json.loads((OUT/'invasion_continuation.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,case,col in zip(axes,['M2','M3'],colors):
        for N,style in [(100,'-'),(200,'--')]:
            rec=continuation[f'{case}_N{N}']['records']
            ax.plot([d['c'] for d in rec],[d['invasion_eigenvalue']*1e6 for d in rec],ls=style,color=col if N==100 else 'black',label=f'N = {N}')
        ax.axhline(0,color='gray',lw=.8)
        crossing=continuation[f'{case}_N200']['crossings']
        for d in crossing: ax.axvline(d['c'],color='gray',ls=':',lw=.9)
        ax.set(xlabel='Common nonlocal coefficient c',ylabel=r'Invasion eigenvalue ($10^{-6}$)',title=f'({"a" if case=="M2" else "b"}) {case}')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ext in ('pdf','png'): fig.savefig(FIG/f'rev2_fig7_invasion.{ext}',dpi=180)
    plt.close(fig)
    plot_holling()
    print('Saved three revision figures.',flush=True)


def plot_holling():
    """Figure 8: maximum-norm errors for Holling II uptake (single panel)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors = ['#185d93','#ba4d26','#418150']
    hdata=json.loads((OUT/'hollingII.json').read_text())
    fig,ax=plt.subplots(figsize=(4.8,3.0),layout='constrained')
    d=hdata['meshes']['200']
    for i,(name,col) in enumerate(zip('uvw',colors)):
        ax.semilogy(d['t'],np.maximum(d['errors'][i],1e-16),color=col,label=name)
    ax.set(xlabel='Time t',ylabel=r'$L^\infty$ error',xlim=(0,35),ylim=(1e-13,1))
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ext in ('pdf','png'): fig.savefig(FIG/f'rev2_fig8_hollingII.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    if mode in ('holling','all'): holling()
    if mode in ('invasion','all'): invasion()
    if mode in ('plots','all'): plots()
