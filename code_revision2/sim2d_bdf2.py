"""2D method of lines with SciPy BDF and an exact sparse Jacobian.

The total masses P = I(u), Q = I(v) are carried as two extra unknowns with
P' = h^2 sum(u k_u), Q' = h^2 sum(v k_v) (the discrete diffusion conserves mass), so that
all couplings are local except two dense rows and two dense columns.  F(w) = w.
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from sim2d import laplacian


def make(N, p):
    h = 1.0 / N; h2 = h * h
    xc = (np.arange(N) + 0.5) * h
    X, Y = np.meshgrid(xc, xc, indexing="xy")
    x, y = X.ravel(), Y.ravel()
    L = laplacian(N)
    r = p["r"](x, y)
    M = N * N
    a1, a2, a3, a4, a5 = (p[k] for k in ("a1", "a2", "a3", "a4", "a5"))
    b1, b2, b3, b4, b5 = (p[k] for k in ("b1", "b2", "b3", "b4", "b5"))
    mu, th1, th2 = p["mu"], p["th1"], p["th2"]
    d1, d2 = p["d1"], p["d2"]
    dd1, dd2 = p["dd1"], p["dd2"]          # derivatives of the motilities

    def split(z):
        return z[:M], z[M:2 * M], z[2 * M:3 * M], z[3 * M], z[3 * M + 1]

    def kin(u, v, w, P, Q):
        ku = a1 * w - a2 * (th1 + u) - a3 * v - a4 * P - a5 * Q
        kv = b1 * w - b2 * u - b3 * (th2 + v) - b4 * Q - b5 * P
        kw = -u - v + mu * (r - w)
        return ku, kv, kw

    def rhs(t, z):
        u, v, w, P, Q = split(z)
        ku, kv, kw = kin(u, v, w, P, Q)
        return np.concatenate((L @ (d1(w) * u) + u * ku, L @ (d2(w) * v) + v * kv, L @ w + w * kw,
                               [h2 * np.sum(u * ku), h2 * np.sum(v * kv)]))

    def jac(t, z):
        u, v, w, P, Q = split(z)
        ku, kv, kw = kin(u, v, w, P, Q)
        D = sp.diags
        Juu = L @ D(d1(w)) + D(ku - a2 * u)
        Juv = D(-a3 * u)
        Juw = L @ D(dd1(w) * u) + D(a1 * u)
        Jvu = D(-b2 * v)
        Jvv = L @ D(d2(w)) + D(kv - b3 * v)
        Jvw = L @ D(dd2(w) * v) + D(b1 * v)
        Jwu = D(-w); Jwv = D(-w)
        Jww = L + D(kw - mu * w)
        colP_u = sp.csr_matrix((-a4 * u).reshape(-1, 1)); colQ_u = sp.csr_matrix((-a5 * u).reshape(-1, 1))
        colP_v = sp.csr_matrix((-b5 * v).reshape(-1, 1)); colQ_v = sp.csr_matrix((-b4 * v).reshape(-1, 1))
        zcol = sp.csr_matrix((M, 1))
        row = lambda a: sp.csr_matrix(np.asarray(a).reshape(1, -1))
        sc = lambda s: sp.csr_matrix([[s]])
        PP = h2 * np.sum(-a4 * u); PQ = h2 * np.sum(-a5 * u)
        QP = h2 * np.sum(-b5 * v); QQ = h2 * np.sum(-b4 * v)
        J = sp.bmat([[Juu, Juv, Juw, colP_u, colQ_u],
                     [Jvu, Jvv, Jvw, colP_v, colQ_v],
                     [Jwu, Jwv, Jww, zcol, zcol],
                     [row(h2 * (ku - a2 * u)), row(h2 * (-a3 * u)), row(h2 * (a1 * u)), sc(PP), sc(PQ)],
                     [row(h2 * (-b2 * v)), row(h2 * (kv - b3 * v)), row(h2 * (b1 * v)), sc(QP), sc(QQ)]],
                    format="csc")
        return J

    return x, y, rhs, jac, h2


def run(N, p, T, t_eval, rtol, atol):
    x, y, rhs, jac, h2 = make(N, p)
    u0, v0, w0 = p["u0"](x, y), p["v0"](x, y), p["w0"](x, y)
    z0 = np.concatenate((u0, v0, w0, [h2 * u0.sum(), h2 * v0.sum()]))
    sol = solve_ivp(rhs, (0.0, T), z0, method="BDF", t_eval=t_eval, rtol=rtol, atol=atol, jac=jac)
    if not sol.success:
        raise RuntimeError(sol.message)
    return x, y, sol
