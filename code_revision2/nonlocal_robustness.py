"""Separate nonlocal self/cross competition and test resource/motility dependence.

Run with BLAS threads set to one before Python starts. See ROBUSTNESS.md.
All new calculations are one-dimensional finite-volume experiments, not PDE proofs.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import json
import platform
import time
import numpy as np
import scipy
from scipy.optimize import root, brentq
from scipy.linalg import eigh_tridiagonal, eigvals
from scipy.integrate import solve_ivp
from sim import make_rhs, base_B

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parent / "figures"
ETAS = [0., .1, .25, .5, .75, .9]
KAPPAS = [0., 1., 1.5, 2., 2.5, 3.]
SELFS = [0., .01, .02, .05, .1, .25, .5]
SIGN_TOL = 2e-9  # Numerical neutral band; mesh comparisons are reported separately.


def save(name, value):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), sign_tolerance=SIGN_TOL)


@lru_cache(None)
def geometry(n):
    x = (np.arange(n) + .5) / n
    L = np.diag(np.full(n, -2. * n*n))
    L += np.diag(np.full(n-1, n*n), 1) + np.diag(np.full(n-1, n*n), -1)
    L[0, 0] = L[-1, -1] = -n*n
    return x, L


def lap(z):
    # Flux differences preserve the original conservative discretization.
    flux = np.diff(z) * len(z)**2
    return np.diff(np.concatenate(([0.], flux, [0.])))


def motility(case, species, w, k):
    responsive = (case == "M2" and species == "v") or (case == "M3" and species == "u")
    prefactor = .1 if species == "u" else 1.
    d = prefactor * np.exp(-k*w) if responsive else np.full_like(w, prefactor)
    return d, -k*d if responsive else np.zeros_like(w)


def resident_system(n, case, species, eta, k, s):
    x, L = geometry(n)
    supply = 1 + eta*np.cos(np.pi*x)

    def fun(z):
        R, W = z[:n], z[n:]
        d, _ = motility(case, species, W, k)
        return np.r_[lap(d*R) + R*(W-R-s*R.mean()), lap(W)+W*(supply-R-W)]

    def jac(z):
        R, W = z[:n], z[n:]
        d, dp = motility(case, species, W, k)
        return np.block([[L*d + np.diag(W-2*R-s*R.mean()) - (s/n)*R[:, None]*np.ones((1,n)),
                          L*(dp*R) + np.diag(R)],
                         [-np.diag(W), L+np.diag(supply-R-2*W)]])
    return fun, jac


def resident(n, case, species, eta, k, s, guess=None):
    fun, jac = resident_system(n, case, species, eta, k, s)
    if guess is None:
        guess = np.r_[np.full(n, 1/(2+s)), np.full(n, (1+s)/(2+s))]
    sol = root(fun, guess, jac=jac, method="hybr", options={"xtol": 2e-11})
    residual = float(np.max(np.abs(fun(sol.x))))
    if residual > 2e-8 or sol.x.min() <= 0:
        raise RuntimeError((n, case, species, eta, k, s, residual, sol.message))
    return sol.x, dict(residual=residual, minimum=float(sol.x.min()), solver_success=bool(sol.success))


def invasion(n, case, absent, z, k, q=0.):
    R, W = z[:n], z[n:]
    d, _ = motility(case, absent, W, k)
    diagonal = -2*n*n*d + W-R-q*R.mean()
    diagonal[[0,-1]] += n*n*d[[0,-1]]
    off = n*n*np.sqrt(d[:-1]*d[1:])
    values, vectors = eigh_tridiagonal(diagonal, off, select="i", select_range=(n-1,n-1),
                                      lapack_driver="stebz", tol=1e-14)
    lam, vec = float(values[0]), vectors[:,0]
    defect = diagonal*vec-lam*vec
    defect[:-1] += off*vec[1:]
    defect[1:] += off*vec[:-1]
    return lam, float(np.max(np.abs(defect)))


def pair(n, case, eta, k, s, guesses=None):
    result = dict(N=n, case=case, eta=eta, kappa=k, self=s)
    states = {}
    for absent, present in (("u","v"), ("v","u")):
        z, diag = resident(n, case, present, eta, k, s,
                           None if guesses is None else guesses[present])
        lam, defect = invasion(n, case, absent, z, k)
        mass = float(z[:n].mean())
        result[absent] = dict(lambda_q0=lam, resident_mass=mass, qcritical=lam/mass,
                              lambda_diagonal=lam-s*mass, eigen_residual=defect, **diag)
        states[present] = z
    return result, states


def classify(lu, lv):
    if abs(lu) <= SIGN_TOL or abs(lv) <= SIGN_TOL:
        return "neutral/unresolved"
    return ("both invade" if lv > 0 else "u only") if lu > 0 else ("v only" if lv > 0 else "neither invades")


def sweep():
    t0 = time.time()
    rows = []
    for n in (100,200):
        for case in ("M2","M3"):
            for eta in ETAS:
                for k in KAPPAS:
                    guess = None
                    for s in SELFS:
                        row, guess = pair(n, case, eta, k, s, guess)
                        row["diagonal_class"] = classify(row["u"]["lambda_diagonal"], row["v"]["lambda_diagonal"])
                        rows.append(row)
                print("sweep", n, case, "eta", eta, flush=True)
    baseline = []
    crossings = []
    for n in (100,200,400):
        for case, rare in (("M2","u"),("M3","v")):
            guess = None
            # A fine self-competition grid resolves the thin region near q=s.
            for s in np.linspace(0,.5,51):
                row, guess = pair(n, case, .5, 2., float(s), guess)
                baseline.append(row)
            def diag_eig(s):
                row, _ = pair(n, case, .5, 2., s)
                return row[rare]["lambda_diagonal"]
            c = brentq(diag_eig, 0., .1, xtol=2e-10)
            crossings.append(dict(N=n, case=case, rare=rare, ccritical=c, lambda_at_root=diag_eig(c)))
            print("baseline",n,case,"crossing",c,flush=True)
    # Independent 400-cell refinements at selected heterogeneous/motility extremes.
    selected = []
    for case in ("M2","M3"):
        for eta,k in ((.1,1.),(.1,2.),(.5,1.5),(.5,2.5),(.9,2.),(.9,3.)):
            for s in (0.,.1,.5):
                row,_=pair(400,case,eta,k,s)
                selected.append(row)
    out = dict(environment=environment(), eta_grid=ETAS, kappa_grid=KAPPAS, self_grid=SELFS,
               rows=rows, baseline=baseline, baseline_crossings=crossings, selected_N400=selected,
               seconds=time.time()-t0)
    save("nonlocal_robustness.json",out)
    return out


def full_system(n, case, eta, k, s, q):
    x,L=geometry(n)
    supply=1+eta*np.cos(np.pi*x)
    def fun(t,y):
        # Algebraically identical to sim.make_rhs. Evaluate the Laplacian
        # as differences of face fluxes to reduce roundoff in the mass mode.
        u,v,w=y[:n],y[n:2*n],y[2*n:]
        du,_=motility(case,"u",w,k); dv,_=motility(case,"v",w,k)
        return np.r_[lap(du*u)+u*(w-u-v-s*u.mean()-q*v.mean()),
                     lap(dv*v)+v*(w-u-v-s*v.mean()-q*u.mean()),
                     lap(w)+w*(supply-u-v-w)]
    def jac(t,y):
        u,v,w=y[:n],y[n:2*n],y[2*n:]
        du,dpu=motility(case,"u",w,k); dv,dpv=motility(case,"v",w,k)
        U=u[:,None]*np.ones((1,n))/n; V=v[:,None]*np.ones((1,n))/n
        return np.block([
            [L*du+np.diag(w-2*u-v-s*u.mean()-q*v.mean())-s*U, -np.diag(u)-q*U, L*(dpu*u)+np.diag(u)],
            [-np.diag(v)-q*V,L*dv+np.diag(w-u-2*v-s*v.mean()-q*u.mean())-s*V,L*(dpv*v)+np.diag(v)],
            [-np.diag(w),-np.diag(w),L+np.diag(supply-u-v-2*w)]])
    return x, fun, jac


def trajectory(n, case, eta, k, s, q, fraction, tight=False):
    x,fun,jac=full_system(n,case,eta,k,s,q)
    y0=np.r_[np.full(n,fraction),np.full(n,1-fraction),1+eta*np.cos(np.pi*x)]
    # The balanced baseline has a much slower rare-species mode than the
    # separated coefficients. Do not misclassify its long transient as steady.
    T=1e7 if s==q else 1e5
    times=np.r_[0.,np.geomspace(.01,T,401)]
    rtol,atol=(1e-10,1e-13) if tight else (1e-9,1e-12)
    sol=solve_ivp(fun,(0,T),y0,method="BDF",jac=jac,t_eval=times,rtol=rtol,atol=atol)
    if not sol.success:raise RuntimeError(sol.message)
    z=sol.y[:,-1]
    stat=root(lambda y:fun(0,y),z,jac=lambda y:jac(0,y),options={"xtol":2e-11})
    stat_res=float(np.max(np.abs(fun(0,stat.x))))
    if stat_res>2e-8 or np.max(np.abs(stat.x-z))>1e-6:
        raise RuntimeError(("stationary mismatch",n,case,eta,k,s,q,fraction,stat_res,
                            float(np.max(np.abs(stat.x-z)))))
    lam=float(eigvals(jac(0,stat.x)).real.max())
    row,_=pair(n,case,eta,k,s)
    lu=row["u"]["lambda_q0"]-q*row["u"]["resident_mass"]
    lv=row["v"]["lambda_q0"]-q*row["v"]["resident_mass"]
    out=dict(N=n,case=case,eta=eta,kappa=k,self=s,cross=q,u_initial_fraction=fraction,
             rtol=rtol,atol=atol,T=T,t=times.tolist(),mass_u=sol.y[:n].mean(axis=0).tolist(),
             mass_v=sol.y[n:2*n].mean(axis=0).tolist(),final_state=z.tolist(),
             final_residual=float(np.max(np.abs(fun(T,z)))),minimum_saved=float(sol.y.min()),
             stationary_residual=stat_res,stationary_distance=float(np.max(np.abs(stat.x-z))),
             stationary_minima=[float(a.min()) for a in stat.x.reshape(3,n)],
             stationary_masses=[float(a.mean()) for a in stat.x.reshape(3,n)],
             leading_eigenvalue=lam,lambda_u=lu,lambda_v=lv,invasion_class=classify(lu,lv),
             nfev=sol.nfev,njev=sol.njev)
    print("trajectory",n,case,eta,k,s,q,fraction,tight,"masses",out["mass_u"][-1],out["mass_v"][-1],flush=True)
    return out


def trajectories():
    t0=time.time(); rows=[]
    design=[]
    for case in ("M2","M3"):
        for s,q,f in ((.1,0.,.5),(0.,.1,.1),(0.,.1,.5),(0.,.1,.9),(.1,.1,.5)):
            design.append((case,.5,2.,s,q,f))
        # Robustness of self/cross separation at weak and strong heterogeneity/motility.
        for eta,k in ((.1,1.),(.9,3.)):
            for s,q,f in ((.1,0.,.5),(0.,.1,.1),(0.,.1,.9)):
                design.append((case,eta,k,s,q,f))
        for eta,k in ((.5,2.5),(.9,3.)):
            design.append((case,eta,k,.5,.5,.5))
    for n in (100,200):
        for args in design:
            rows.append(trajectory(n,*args))
            save("nonlocal_trajectories.json",dict(environment=environment(),rows=rows,
                 refinements=[],complete=False,seconds=time.time()-t0))
    # Time-tolerance and 400-cell checks for all three baseline coefficient choices.
    refinements=[]
    for case in ("M2","M3"):
        for s,q,f in ((.1,0.,.5),(0.,.1,.1),(0.,.1,.9),(.1,.1,.5)):
            refinements.append(trajectory(200,case,.5,2.,s,q,f,tight=True))
            refinements.append(trajectory(400,case,.5,2.,s,q,f,tight=True))
    out=dict(environment=environment(),rows=rows,refinements=refinements,complete=True,seconds=time.time()-t0)
    save("nonlocal_trajectories.json",out)
    return out


def checks():
    rng=np.random.default_rng(20261018)
    report=dict(environment=environment())
    hom=[]
    for n in (100,200,400):
        for case in ("M2","M3"):
            for k in (0.,2.,3.):
                for s,q in ((0.,0.),(.1,0.),(0.,.1),(.1,.1),(.5,.1)):
                    row,states=pair(n,case,0.,k,s)
                    for absent,present in (("u","v"),("v","u")):
                        z=states[present]
                        actual,_=invasion(n,case,absent,z,k,q)
                        expected=(s-q)/(2+s)
                        hom.append(abs(actual-expected))
    report["homogeneous_max_eigenvalue_error"]=max(hom)
    assert max(hom)<1e-9
    shift=[]; similarity=[]; jacobian=[]; mass=[]; resident_jac=[]; original_rhs=[]
    for case in ("M2","M3"):
        n=30; eta=.75;k=2.5;s=.1;q=.07
        row,states=pair(n,case,eta,k,s)
        for absent,present in (("u","v"),("v","u")):
            z=states[present];R,W=z[:n],z[n:]; _,L=geometry(n)
            computed,_=invasion(n,case,absent,z,k,q)
            shifted=row[absent]["lambda_q0"]-q*R.mean()
            shift.append(abs(computed-shifted))
            d,_=motility(case,absent,W,k)
            direct=float(eigvals(L*d+np.diag(W-R-q*R.mean())).real.max())
            similarity.append(abs(computed-direct))
            rf,rj=resident_system(n,case,present,eta,k,s)
            direction=rng.normal(size=2*n); eps=1e-6
            exact=rj(z)@direction
            fd=(rf(z+eps*direction)-rf(z-eps*direction))/(2*eps)
            resident_jac.append(float(np.max(np.abs(exact-fd))/np.max(np.abs(exact))))
        x,fun,jac=full_system(n,case,eta,k,s,q)
        y=np.r_[.3+.05*np.cos(np.pi*x),.2+.02*np.sin(np.pi*x),.6+.1*np.cos(np.pi*x)]
        direction=rng.normal(size=3*n);eps=1e-6
        exact=jac(0,y)@direction
        fd=(fun(0,y+eps*direction)-fun(0,y-eps*direction))/(2*eps)
        jacobian.append(float(np.max(np.abs(exact-fd))/np.max(np.abs(exact))))
        u,v,w=y[:n],y[n:2*n],y[2*n:]
        expected=[np.mean(u*(w-u-v-s*u.mean()-q*v.mean())),np.mean(v*(w-u-v-s*v.mean()-q*u.mean())),np.mean(w*(1+eta*np.cos(np.pi*x)-u-v-w))]
        actual=fun(0,y).reshape(3,n).mean(axis=1)
        mass.append(float(np.max(np.abs(actual-expected))))
        p=base_B(False)
        p.update(a4=s,b4=s,a5=q,b5=q,r=lambda x:1+eta*np.cos(np.pi*x),
                 d1=lambda w:motility(case,"u",w,k)[0],d2=lambda w:motility(case,"v",w,k)[0])
        _,legacy=make_rhs(n,p)
        original_rhs.append(float(np.max(np.abs(fun(0,y)-legacy(0,y)))))
    report.update(spectral_shift_max_error=max(shift),similarity_max_error=max(similarity),
                  resident_jacobian_relative_error=max(resident_jac),full_jacobian_relative_error=max(jacobian),
                  noflux_mass_balance_max_defect=max(mass),original_rhs_max_difference=max(original_rhs))
    assert max(shift+similarity)<1e-9
    assert max(resident_jac+jacobian)<1e-7
    assert max(mass)<1e-10
    assert max(original_rhs)<1e-10
    sw=json.loads((RESULTS/"nonlocal_robustness.json").read_text())
    def key(r):return (r["case"],r["eta"],r["kappa"],r["self"])
    coarse={key(r):r for r in sw["rows"] if r["N"]==100}
    fine={key(r):r for r in sw["rows"] if r["N"]==200}
    pairs=[(coarse[k],fine[k]) for k in coarse]
    report["mesh_100_200"]=dict(
        max_eigenvalue_difference=max(abs(a[t]["lambda_diagonal"]-b[t]["lambda_diagonal"]) for a,b in pairs for t in "uv"),
        max_qcritical_difference=max(abs(a[t]["qcritical"]-b[t]["qcritical"]) for a,b in pairs for t in "uv"),
        class_disagreements=sum(a["diagonal_class"]!=b["diagonal_class"] for a,b in pairs),
        paired_parameter_sets=len(pairs))
    pairs4=[(fine[key(r)],r) for r in sw["selected_N400"]]
    report["mesh_200_400_selected"]=dict(
        max_eigenvalue_difference=max(abs(a[t]["lambda_diagonal"]-b[t]["lambda_diagonal"]) for a,b in pairs4 for t in "uv"),
        max_qcritical_difference=max(abs(a[t]["qcritical"]-b[t]["qcritical"]) for a,b in pairs4 for t in "uv"),
        class_disagreements=sum(a["diagonal_class"]!=classify(b["u"]["lambda_diagonal"],b["v"]["lambda_diagonal"]) for a,b in pairs4),
        paired_parameter_sets=len(pairs4))
    allrows=sw["rows"]+sw["baseline"]+sw["selected_N400"]
    report["max_resident_residual"]=max(r[t]["residual"] for r in allrows for t in "uv")
    report["max_symmetric_eigen_residual"]=max(r[t]["eigen_residual"] for r in allrows for t in "uv")
    old=json.loads((RESULTS/"invasion_continuation.json").read_text())
    report["baseline_crossings"]=sw["baseline_crossings"]
    baseline_differences=[]
    for r in sw["baseline"]:
        if r["N"]==400:continue
        rare="u" if r["case"]=="M2" else "v"
        for archived in old[f"{r['case']}_N{r['N']}"]["records"]:
            if abs(r["self"]-archived["c"])<1e-12:
                baseline_differences.append(abs(r[rare]["lambda_diagonal"]-archived["invasion_eigenvalue"]))
    report["archived_baseline_max_eigenvalue_difference"]=max(baseline_differences)
    assert max(baseline_differences)<1e-9
    report["archived_baseline_max_crossing_difference"]=max(
        abs(r["ccritical"]-old[f"{r['case']}_N{r['N']}"]["crossings"][0]["c"])
        for r in sw["baseline_crossings"] if r["N"]!=400)
    assert report["archived_baseline_max_crossing_difference"]<5e-7
    tr=json.loads((RESULTS/"nonlocal_trajectories.json").read_text())
    assert tr["complete"]
    def tk(r):return (r["case"],r["eta"],r["kappa"],r["self"],r["cross"],r["u_initial_fraction"])
    a={tk(r):r for r in tr["rows"] if r["N"]==100}
    b={tk(r):r for r in tr["rows"] if r["N"]==200}
    def mdiff(r1,r2):return max(abs(r1["mass_"+t][-1]-r2["mass_"+t][-1]) for t in "uv")
    report["trajectories_100_200_max_final_mass_difference"]=max(mdiff(a[k],b[k]) for k in a)
    tight={tk(r):r for r in tr["refinements"] if r["N"]==200}
    fin={tk(r):r for r in tr["refinements"] if r["N"]==400}
    report["trajectories_tight_tolerance_max_final_mass_difference"]=max(mdiff(b[k],tight[k]) for k in tight)
    report["trajectories_tight_tolerance_max_mass_history_difference"]=max(
        float(np.max(np.abs(np.array(b[k]['mass_'+t])-np.array(tight[k]['mass_'+t]))))
        for k in tight for t in 'uv')
    report["trajectories_200_400_max_final_mass_difference"]=max(mdiff(tight[k],fin[k]) for k in fin)
    report["trajectories_max_final_residual"]=max(r["final_residual"] for r in tr["rows"]+tr["refinements"])
    report["trajectories_max_stationary_distance"]=max(r["stationary_distance"] for r in tr["rows"]+tr["refinements"])
    report["trajectories_minimum_saved"]=min(r["minimum_saved"] for r in tr["rows"]+tr["refinements"])
    report["trajectories_largest_leading_eigenvalue"]=max(r["leading_eigenvalue"] for r in tr["rows"]+tr["refinements"])
    report["status"]="passed"
    assert report["mesh_100_200"]["class_disagreements"]==0
    assert report["mesh_200_400_selected"]["class_disagreements"]==0
    assert report["trajectories_tight_tolerance_max_final_mass_difference"]<1e-7
    assert report["trajectories_tight_tolerance_max_mass_history_difference"]<1e-6
    assert report["trajectories_minimum_saved"]>-1e-9
    assert report["trajectories_largest_leading_eigenvalue"]<0
    save("nonlocal_checks.json",report)
    print(json.dumps(report,indent=2),flush=True)


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"ps.fonttype":42})
    sw=json.loads((RESULTS/"nonlocal_robustness.json").read_text())
    tr=json.loads((RESULTS/"nonlocal_trajectories.json").read_text())
    colors={"u":"#0072B2","v":"#D55E00"}
    fig,axs=plt.subplots(1,2,figsize=(10.4,3.7),layout="constrained")
    for ax,case in zip(axs,("M2","M3")):
        rows=[r for r in sw["baseline"] if r["N"]==400 and r["case"]==case]
        s=np.array([r["self"] for r in rows])
        for t in "uv":
            ax.plot(s,[1e3*(r[t]["qcritical"]-r["self"]) for r in rows],color=colors[t],label=f"{t} invasion threshold")
        ax.axhline(0,color=".5",ls=":",label="$q=s$")
        ax.set(xlabel="Nonlocal self-competition $s$",ylabel="$10^3(q_{\\mathrm{crit}}-s)$",title=case+": threshold relative to $q=s$")
        ax.legend(frameon=False,fontsize=8)
    for ext in ("pdf","png"):fig.savefig(FIGURES/f"rev3_fig9_separation.{ext}",dpi=180)
    plt.close(fig)
    # Markers show sampled parameter combinations only (no interpolated phase boundary).
    cats={"both invade":("#009E73","o"),"u only":("#0072B2",">"),"v only":("#D55E00","<"),"neither invades":("#CC79A7","s"),"neutral/unresolved":("#777777","x")}
    fig,axs=plt.subplots(2,2,figsize=(9.3,6.4),layout="constrained")
    for i,case in enumerate(("M2","M3")):
        for j,s in enumerate((0.,.5)):
            ax=axs[i,j]
            for cat,(col,marker) in cats.items():
                rows=[r for r in sw["rows"] if r["N"]==200 and r["case"]==case and r["self"]==s and r["diagonal_class"]==cat]
                ax.scatter([r["kappa"] for r in rows],[r["eta"] for r in rows],c=col,marker=marker,s=65)
            ax.set(xlabel="Motility sensitivity $\\kappa$",ylabel="Resource heterogeneity $\\eta$",title=f"{case}, $s=q={s:g}$",xticks=KAPPAS,yticks=ETAS,xlim=(-.12,3.15),ylim=(-.04,.96))
            ax.grid(alpha=.16)
    handles=[Line2D([],[],color=col,marker=m,ls="",label=cat) for cat,(col,m) in cats.items()]
    fig.legend(handles=handles,loc="outside lower center",ncol=5,fontsize=8)
    for ext in ("pdf","png"):fig.savefig(FIGURES/f"rev3_fig10_robustness.{ext}",dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10.2,6.2),layout="constrained")
    for i,case in enumerate(("M2","M3")):
        for j,(s,q) in enumerate(((.1,0.),(0.,.1))):
            ax=axs[i,j]
            for frac in ((.1,.9) if q else (.5,)):
                row=next(r for r in tr["rows"] if r["N"]==200 and r["case"]==case and r["eta"]==.5 and r["kappa"]==2 and r["self"]==s and r["cross"]==q and r["u_initial_fraction"]==frac)
                if q:
                    u=np.array(row["mass_u"]);v=np.array(row["mass_v"])
                    ax.semilogx(row["t"][1:],(u/(u+v))[1:],color="#0072B2" if frac==.1 else "#9A3C82",
                                ls="-" if frac==.1 else "--",label=rf"$u_0={frac:g},\ v_0={1-frac:g}$")
                else:
                    for t,style in (("u","-"),("v","--")):
                        ax.semilogx(row["t"][1:],row["mass_"+t][1:],color=colors[t],ls=style,label=f"{t}, $u_0={frac:g}$")
            ax.set(xlabel="Time",ylabel="$u$ share of consumer mass" if q else "Total mass",
                   title=rf"{case}: $s={s:g},\ q={q:g}$",ylim=(-.03,1.03) if q else (-.015,.55))
            ax.legend(frameon=False,fontsize=8)
    for ext in ("pdf","png"):fig.savefig(FIGURES/f"rev3_fig11_trajectories.{ext}",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["pilot","sweep","trajectories","checks","plots","all"])
    args=parser.parse_args()
    if args.mode=="pilot":
        for case in ("M2","M3"):
            for s in (0.,.1,.5):
                row,_=pair(100,case,.5,2.,s)
                print(json.dumps(row),flush=True)
    else:
        modes=("sweep","trajectories","checks","plots") if args.mode=="all" else (args.mode,)
        for mode in modes:globals()[mode]()
