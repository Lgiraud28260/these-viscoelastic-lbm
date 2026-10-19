#!/usr/bin/env python3
"""Vérification numérique : c_s^2 est-il bien un paramètre libre du modèle D2Q9
du chapitre 3 (Part I, « Construction of the Models ») ?

Le modèle est codé tel qu'il est décrit dans la thèse :
  - base orthogonale des 9 moments (1, E, Cx, Cy, Txxy, Tyyx, Qxx, Qxy, H) ;
  - matrice de collision diagonale dans cette base, de valeurs propres
    (0, λe, 0, 0, λQ, λQ, λν, λν, λσ) ;
  - équilibres N0 = ((3-5cs²)/3)(ρ - 2 j²/((3-5cs²)ρ)),
              Ni = t_p (ρ + j·c/cs² + j_α j_β (3 c_α c_β - δ_αβ)/(2ρ cs²)),
    avec t_1 = cs²/3 et t_2 = cs²/12.

Deux mesures indépendantes de c_s² pour chaque valeur imposée :
  1. dynamique : fréquence d'une onde sonore stationnaire,
     ρ(x,t) = ρ0 + δρ cos(kx) e^{-Γt} cos(ωt), ω² = c_s² k² - Γ² ;
  2. thermodynamique : profil hydrostatique sous une force de volume F
     entre deux parois (rebond), c_s² dρ/dy = ρ F  =>  ρ ∝ exp(F y / c_s²).

Usage : python3 experiences/mesure_cs2.py   (≈ 1 min ; écrit les figures cs2-*.eps/png
 et les valeurs mesurées dans mesure_cs2.json)
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

# Vitesses dans l'ordre des vecteurs propres de la thèse : repos, axes, diagonales
C = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)])
CX, CY = C[:, 0], C[:, 1]
C2 = CX**2 + CY**2
OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]

# Base orthogonale (équation « vectorlist » du chapitre 3)
M = np.array([
    np.ones(9),                       # 1   : masse
    4 - 3 * C2,                       # E   : énergie
    CX, CY,                           # Cx, Cy : quantité de mouvement
    2 * CX**3 - 3 * CY**2 * CX,       # Txxy : flux d'énergie q_x
    2 * CY**3 - 3 * CX**2 * CY,       # Tyyx : flux d'énergie q_y
    CX**2 - C2 / 2,                   # Qxx : contrainte
    CX * CY,                          # Qxy : contrainte
    9 * CX**2 * CY**2 - 6 * C2 + 4,   # H
], dtype=float)
M_INV = np.linalg.inv(M)


def equilibrium(rho, jx, jy, cs2):
    """Populations d'équilibre (fin du chapitre 3), tableaux de forme (9, ...)."""
    tp = np.where(C2 == 1, cs2 / 3, cs2 / 12)
    j2 = jx**2 + jy**2
    neq = np.empty((9,) + rho.shape)
    for i in range(1, 9):
        cj = CX[i] * jx + CY[i] * jy
        quad = (3 * CX[i]**2 - 1) * jx**2 + (3 * CY[i]**2 - 1) * jy**2 + 6 * CX[i] * CY[i] * jx * jy
        neq[i] = tp[i] * (rho + cj / cs2 + quad / (2 * rho * cs2))
    neq[0] = (3 - 5 * cs2) / 3 * (rho - 2 * j2 / ((3 - 5 * cs2) * rho))
    return neq


class LBM:
    def __init__(self, nx, ny, cs2, lam_nu=1.2, lam_e=1.0, lam_q=1.2, lam_s=1.2,
                 force=0.0, walls=False):
        self.nx, self.ny, self.cs2 = nx, ny, cs2
        lam = np.diag([0, lam_e, 0, 0, lam_q, lam_q, lam_nu, lam_nu, lam_s])
        self.omega = M_INV @ lam @ M             # Ω = T⁻¹ Ω̂ T
        self.force, self.walls = force, walls
        self.solid = np.zeros((nx, ny), bool)
        if walls:                                # parois en y = 0 et y = ny-1
            self.solid[:, 0] = self.solid[:, -1] = True
        self.nu = (1 / lam_nu - 0.5) / 3
        self.xi = (2 / 3 - cs2) * (1 / lam_e - 0.5)

    def macros(self, n):
        rho = n.sum(0)
        jx = np.tensordot(CX, n, 1)
        jy = np.tensordot(CY, n, 1) + 0.5 * rho * self.force   # j redéfini avec la force (ch. 4)
        return rho, jx, jy

    def step(self, n):
        rho, jx, jy = self.macros(n)
        neq = equilibrium(rho, jx, jy, self.cs2)
        post = n - np.tensordot(self.omega, n - neq, 1)
        if self.force:
            tp = np.where(C2 == 1, self.cs2 / 3, self.cs2 / 12)
            post += (tp * CY / self.cs2)[:, None, None] * rho * self.force
        if self.walls:                           # rebond (bounce-back) sur les nœuds solides
            post[:, self.solid] = n[OPP][:, self.solid]
        for i in range(9):
            post[i] = np.roll(np.roll(post[i], CX[i], 0), CY[i], 1)
        return post


def sound_wave(cs2, nx=200, steps=None, amp=1e-3):
    """Mesure dynamique : onde sonore stationnaire dans un domaine périodique."""
    sim = LBM(nx, 1, cs2)
    k = 2 * np.pi / nx
    if steps is None:                            # environ cinq périodes acoustiques
        steps = int(5 * nx / np.sqrt(cs2))
    x = np.arange(nx)[:, None]
    rho = 1 + amp * np.cos(k * x)
    n = equilibrium(rho, 0 * rho, 0 * rho, cs2)
    t, signal = np.arange(steps), np.empty(steps)
    for s in range(steps):
        signal[s] = (n.sum(0)[:, 0] - 1) @ np.cos(k * x[:, 0]) * 2 / nx
        n = sim.step(n)
    model = lambda t, a, g, w, p: a * np.exp(-g * t) * np.cos(w * t + p)
    w0 = np.sqrt(cs2) * k
    (a, g, w, p), _ = curve_fit(model, t, signal, p0=(amp, (sim.nu + sim.xi) * k**2 / 2, w0, 0))
    # ω² = c_s² k² - Γ²  (relation de dispersion amortie). La correction de réseau
    # en O(k²) est négligeable pour nx = 200 (k ≈ 0.03), voir dispersion().
    return (w**2 + g**2) / k**2, (t, signal, model(t, a, g, w, p))


def hydrostatic(cs2, ny=60, force=2e-4, steps=40000, history_every=0):
    """Mesure thermodynamique : équilibre hydrostatique sous une force F (axe y).
    Les ondes sonores piégées entre les parois s'amortissent lentement : on itère
    longtemps et on vérifie que le fluide est bien au repos.
    Si history_every > 0, renvoie aussi l'estimation de c_s² au cours du temps."""
    sim = LBM(1, ny, cs2, force=-force, walls=True)
    rho = np.ones((1, ny))
    n = equilibrium(rho, 0 * rho, 0 * rho, cs2)
    y = np.arange(3, ny - 3)                     # on écarte les nœuds proches des parois

    def estimate(n):
        slope = np.polyfit(y, np.log(n.sum(0)[0, 3:-3]), 1)[0]   # d ln ρ / dy = -F / c_s²
        return force / -slope

    history = []
    for s in range(1, steps + 1):
        n = sim.step(n)
        if history_every and s % history_every == 0:
            _, _, jy = sim.macros(n)
            history.append((s, estimate(n), np.abs(jy[0, 1:-1]).max() / (force * ny)))
    _, _, jy = sim.macros(n)
    residual = np.abs(jy[0, 1:-1]).max() / (force * ny)
    assert residual < 1e-3, "équilibre hydrostatique non atteint (|j| résiduel %.1e)" % residual
    return estimate(n), (y, n.sum(0)[0, 3:-3]), history


def dispersion(cs2, sizes=(400, 200, 100, 50, 25, 16, 10)):
    """c_s² apparent en fonction du nombre d'onde k = 2π/nx (effets de réseau)."""
    return [(2 * np.pi / nx, sound_wave(cs2, nx=nx)[0]) for nx in sizes]


def figures(values, rows, waves, profiles, disp, history, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(values)))
    saved = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("eps", "png", "svg"):
            fig.savefig(os.path.join(outdir, "%s.%s" % (name, ext)), dpi=130)
        saved.append(name)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for c, cs2 in zip(colors, values):
        t, sig, _ = waves[cs2]
        ax.plot(t, sig / sig[0], color=c, lw=1, label="$c_s^2$ = %.3f" % cs2)
    ax.set(xlabel="time step", ylabel=r"$\hat\rho_k(t)\,/\,\hat\rho_k(0)$", xlim=(0, 3000))
    ax.legend(fontsize=9, loc="lower left")
    save(fig, "cs2-sound-wave")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for c, cs2 in zip(colors, values):
        y, prof = profiles[cs2]
        ax.plot(y, np.log(prof / prof[0]), "o", ms=3, color=c, label="$c_s^2$ = %.3f" % cs2)
    ax.set(xlabel="$y$ (lattice units)", ylabel=r"$\ln\left(\rho(y)/\rho(y_0)\right)$")
    ax.legend(fontsize=9)
    save(fig, "cs2-hydrostatic")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0.15, 0.55], [0.15, 0.55], color="0.6", lw=1, label="measured = imposed")
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o", ms=8, label="sound wave (dynamics)")
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "x", ms=10, mew=2, label="hydrostatics (equation of state)")
    ax.set(xlabel="imposed $c_s^2$", ylabel="measured $c_s^2$")
    ax.legend(fontsize=9)
    save(fig, "cs2-measured-vs-imposed")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for cs2, pts in disp.items():
        k = np.array([p[0] for p in pts])
        ax.plot(k**2, [100 * (p[1] / cs2 - 1) for p in pts], "o-", label="$c_s^2$ = %.3f" % cs2)
    ax.axhline(0, color="0.6", lw=1)
    ax.set(xlabel="$k^2$", ylabel=r"$(c_{s,\rm app}^2 - c_s^2)/c_s^2$  (%)")
    ax.legend(fontsize=9)
    save(fig, "cs2-dispersion")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    s = np.array([h[0] for h in history])
    cs2_h = 0.4
    ax.semilogy(s, [abs(h[1] / cs2_h - 1) for h in history], label="relative error on $c_s^2$")
    ax.semilogy(s, [h[2] for h in history], label=r"residual $\max|j_y|\,/\,(F\,L)$")
    ax.axvline(6000, color="0.5", ls="--", lw=1)
    ax.set(xlabel="time step", ylim=(1e-7, 2))
    ax.legend(fontsize=9)
    save(fig, "cs2-hydrostatic-convergence")
    return saved


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    values = [0.20, 0.25, 1 / 3, 0.40, 0.50]
    rows, waves, profiles = [], {}, {}
    for cs2 in values:
        dyn, waves[cs2] = sound_wave(cs2)
        thermo, profiles[cs2], _ = hydrostatic(cs2)
        rows.append((cs2, dyn, thermo))
        print("c_s² imposé %.4f | onde sonore %.5f (écart %+.3f %%) | hydrostatique %.5f (écart %+.4f %%)"
              % (cs2, dyn, 100 * (dyn / cs2 - 1), thermo, 100 * (thermo / cs2 - 1)), flush=True)

    disp = {cs2: dispersion(cs2) for cs2 in (0.20, 1 / 3, 0.50)}
    for cs2, pts in disp.items():
        print("dispersion c_s² = %.3f :" % cs2, ", ".join("k=%.3f→%+.2f %%" % (k, 100 * (v / cs2 - 1)) for k, v in pts))

    _, _, history = hydrostatic(0.4, history_every=200)
    for s, est, res in history:
        if s in (2000, 6000, 10000, 20000, 40000):
            print("hydrostatique c_s²=0.4, pas %5d : estimation %.4f, résidu %.1e" % (s, est, res))

    with open(os.path.join(outdir, "mesure_cs2.json"), "w") as f:
        json.dump({"table": rows,
                   "dispersion": {"%.4f" % k: v for k, v in disp.items()},
                   "history_cs2_0.4": history}, f, indent=1)
    print("figures :", figures(values, rows, waves, profiles, disp, history, outdir))


if __name__ == "__main__":
    main()
