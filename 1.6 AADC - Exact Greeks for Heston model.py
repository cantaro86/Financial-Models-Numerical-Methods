#!/usr/bin/env python3
"""
1.6 AADC — Exact Greeks for Heston model

This example shows how AADC (Automatic Adjoint Differentiation) can compute
exact Greeks of Heston-priced European options in a single reverse pass,
replacing finite differences entirely.

Four levels of speedup are demonstrated:

  A) Fourier pricing on AADC tape (PROJ method, Kirkby 2015)
     - Price + 5 model Greeks in ~1 ms  (vs ~150 ms FD)

  B) Calibration with exact Jacobian
     - Least-squares fit to 20 instruments

  C) AITF (Adjoint Implicit Function Theorem)
     - d(calibrated params)/d(market prices) via one Jacobian + 5x5 solve
     - vs 40 recalibrations with FD

  D) Monte Carlo on AADC tape (EulerLog scheme, batch API)
     - Price + 6 Greeks (including Delta) in one batch call
     - Honest comparison against FMNM's own Cython MC

All results verified against the library's own Fourier_inversion()
and against finite differences.

Requirements:
  pip install aadc    # free evaluation, matlogica.com/aadc
"""
import numpy as np
import time

from FMNM.Parameters import Option_param
from FMNM.Processes import Heston_process
from FMNM.Heston_pricer import Heston_pricer

try:
    import aadc
    from aadc import idouble
except ImportError:
    raise ImportError(
        "This example requires AADC.\n"
        "Install: pip install aadc   (free evaluation, no licence key)\n"
        "See: https://matlogica.com/aadc"
    )


# ── Complex idouble helpers ──────────────────────────────────────────────────

def _cmul(ar, ai, br, bi):
    return ar*br - ai*bi, ar*bi + ai*br

def _cdiv(ar, ai, br, bi):
    d = br*br + bi*bi + 1e-300
    return (ar*br + ai*bi)/d, (ai*br - ar*bi)/d

def _cexp(ar, ai):
    e = aadc.math.exp(ar)
    return e*aadc.math.cos(ai), e*aadc.math.sin(ai)

def _csqrt_pos(ar, ai):
    m = aadc.math.sqrt(ar*ar + ai*ai + 1e-300)
    s = aadc.math.sqrt((m + ar)*idouble(0.5) + 1e-300)
    return s, ai/(s*idouble(2.0) + 1e-300)

def _clog_pos(ar, ai):
    m = aadc.math.sqrt(ar*ar + ai*ai + 1e-300)
    return aadc.math.log(m + 1e-300), aadc.math.atan(ai/(ar + 1e-300))


# ── Heston CF on AADC tape ───────────────────────────────────────────────────

def _heston_cf(w_arr, v0, theta, kappa, xi, rho, T, drift):
    phi_re, phi_im = [], []
    for w in w_arr:
        a_re, a_im = idouble(-0.5*w*w), idouble(-0.5*w)
        b_re, b_im = kappa, idouble(0.0) - rho*xi*idouble(w)
        o2 = xi*xi
        b2r, b2i = _cmul(b_re, b_im, b_re, b_im)
        agr, agi = _cmul(a_re, a_im, o2*idouble(0.5), idouble(0.0))
        D_re, D_im = _csqrt_pos(b2r - agr*idouble(4.0), b2i - agi*idouble(4.0))
        bDr, bDi = b_re - D_re, b_im - D_im
        eDr, eDi = _cexp(D_re*idouble(-T), D_im*idouble(-T))
        Gr, Gi = _cdiv(bDr, bDi, b_re + D_re, b_im + D_im)
        bDo2r, bDo2i = _cdiv(bDr, bDi, o2, idouble(0.0))
        GeDr, GeDi = _cmul(Gr, Gi, eDr, eDi)
        nr, ni = idouble(1.0) - eDr, idouble(0.0) - eDi
        dr, di = idouble(1.0) - GeDr, idouble(0.0) - GeDi
        fr, fi = _cdiv(nr, ni, dr, di)
        Br, Bi = _cmul(bDo2r, bDo2i, fr, fi)
        psir, psii = _cdiv(dr, di, idouble(1.0) - Gr, idouble(0.0) - Gi)
        kto2r, kto2i = _cdiv(kappa*theta, idouble(0.0), o2, idouble(0.0))
        lpr, lpi = _clog_pos(psir, psii)
        ir_, ii_ = bDr*idouble(T) - lpr*idouble(2.0), bDi*idouble(T) - lpi*idouble(2.0)
        Ar, Ai = _cmul(kto2r, kto2i, ir_, ii_)
        er = Ar + Br*v0
        ei = Ai + Bi*v0 + idouble(w*drift*T)
        pr, pi = _cexp(er, ei)
        phi_re.append(pr); phi_im.append(pi)
    return phi_re, phi_im


def _fft(xr, xi):
    N = len(xr)
    bits = int(np.log2(N))
    def rev(x, b):
        r = 0
        for _ in range(b): r = (r<<1)|(x&1); x >>= 1
        return r
    yr = [xr[rev(i, bits)] for i in range(N)]
    yi = [xi[rev(i, bits)] for i in range(N)]
    length = 2
    while length <= N:
        half = length // 2
        ang = -2.0*np.pi/length
        twr = [np.cos(ang*k) for k in range(half)]
        twi = [np.sin(ang*k) for k in range(half)]
        for s in range(0, N, length):
            for k in range(half):
                i0, i1 = s+k, s+k+half
                tr = yr[i1]*idouble(twr[k]) - yi[i1]*idouble(twi[k])
                ti = yr[i1]*idouble(twi[k]) + yi[i1]*idouble(twr[k])
                yr[i1], yi[i1] = yr[i0]-tr, yi[i0]-ti
                yr[i0], yi[i0] = yr[i0]+tr, yi[i0]+ti
        length *= 2
    return yr, yi


# ── PROJ grid + tape pricing ─────────────────────────────────────────────────

def _proj_grid(S0, K, T, r, v0_d, theta_d, kappa_d, xi_d, rho_d, N):
    lws = np.log(K/S0)
    drift = (r) - 0.5*theta_d
    c1 = T*drift + (1 - np.exp(-kappa_d*T))*(theta_d - v0_d)/(2*kappa_d)
    c2 = 1/(8*kappa_d**3)*(
        xi_d*T*kappa_d*np.exp(-kappa_d*T)*(v0_d-theta_d)*(8*kappa_d*rho_d-4*xi_d)
        + kappa_d*rho_d*xi_d*(1-np.exp(-kappa_d*T))*(16*theta_d-8*v0_d)
        + 2*theta_d*kappa_d*T*(-4*kappa_d*rho_d*xi_d+xi_d**2+4*kappa_d**2)
        + xi_d**2*((theta_d-2*v0_d)*np.exp(-2*kappa_d*T)+theta_d*(6*np.exp(-kappa_d*T)-7)+2*v0_d)
        + 8*kappa_d**2*(v0_d-theta_d)*(1-np.exp(-kappa_d*T)))
    alph = max(10.0*np.sqrt(abs(c2)), 1.15*abs(lws)+c1)
    dx = 2*alph/(N-1); a = 1.0/dx
    lam = c1 - (N/2-1)*dx
    nbar = min(int(np.floor(a*(lws-lam)+1)), N-1)
    xmin = lws - (nbar-1)*dx
    dw = 2*np.pi/(N*dx); w_nz = dw*np.arange(1, N)
    b0, b1, b2c, b3 = 1208/2520, 1191/2520, 120/2520, 1/2520
    zeta = (np.sin(w_nz/(2*a))/w_nz)**4/(b0+b1*np.cos(w_nz/a)+b2c*np.cos(2*w_nz/a)+b3*np.cos(3*w_nz/a))
    phase = np.exp(-1j*xmin*dw*np.arange(0, N))
    cons = 32*a**4
    g1 = 1/24-1/20*np.exp(dx)*(np.exp(-7/4*dx)/54+np.exp(-1.5*dx)/18+np.exp(-1.25*dx)/2+7*np.exp(-dx)/27)
    g2 = .5-.05*(28/27+np.exp(-7/4*dx)/54+np.exp(-1.5*dx)/18+np.exp(-1.25*dx)/2
                 +14*np.exp(-dx)/27+121/54*np.exp(-.75*dx)+23/18*np.exp(-.5*dx)+235/54*np.exp(-.25*dx))
    g3 = 23/24-np.exp(-dx)/90*((28+7*np.exp(-dx))/3+(14*np.exp(dx)+np.exp(-7/4*dx)
         +242*np.cosh(.75*dx)+470*np.cosh(.25*dx))/12+.25*(np.exp(-1.5*dx)+9*np.exp(-1.25*dx)+46*np.cosh(.5*dx)))
    g4 = 14/3*(2+np.cosh(dx))+.5*(np.cosh(1.5*dx)+9*np.cosh(1.25*dx)+23*np.cosh(.5*dx)) \
         +1/6*(np.cosh(7/4*dx)+121*np.cosh(.75*dx)+235*np.cosh(.25*dx))
    G = np.zeros(nbar+1)
    G[nbar], G[nbar-1], G[nbar-2] = K*g1, K*g2, K*g3
    G[:nbar-2] = K - S0*np.exp(xmin)*np.exp(dx*np.arange(0, nbar-2))/90*g4
    disc = np.exp(-r*T)
    return dict(N=N, w_nz=w_nz, zeta=zeta, phase=phase, cons=cons,
                nbar=nbar, G=G, scale=cons*disc/N, drift=r,
                parity=(S0*np.exp(r*T)-K)*disc)


def _proj_on_tape(v0, theta, kappa, xi, rho, T, gp):
    N = gp['N']
    phi_re, phi_im = _heston_cf(gp['w_nz'], v0, theta, kappa, xi, rho, T, gp['drift'])
    gr = [idouble(1.0/gp['cons'])] + [None]*(N-1)
    gi = [idouble(0.0)] + [None]*(N-1)
    zeta, phase = gp['zeta'], gp['phase']
    for k in range(1, N):
        zr, zi = float(np.real(zeta[k-1])), float(np.imag(zeta[k-1]))
        pr = phi_re[k-1]*idouble(zr) - phi_im[k-1]*idouble(zi)
        pi_ = phi_re[k-1]*idouble(zi) + phi_im[k-1]*idouble(zr)
        phr, phi = float(np.real(phase[k])), float(np.imag(phase[k]))
        gr[k] = pr*idouble(phr) - pi_*idouble(phi)
        gi[k] = pr*idouble(phi) + pi_*idouble(phr)
    beta_re, _ = _fft(gr, gi)
    price = idouble(0.0)
    for j in range(gp['nbar']+1):
        price = price + beta_re[j]*idouble(gp['G'][j])
    return price*idouble(gp['scale']) + idouble(gp['parity'])


def record_fourier(S0, K, T, r, v0, theta, kappa, xi, rho, N=512):
    """Record PROJ Fourier pricing on AADC tape."""
    gp = _proj_grid(S0, K, T, r, v0, theta, kappa, xi, rho, N)
    fn = aadc.Functions()
    pnames = ['v0', 'theta', 'kappa', 'xi', 'rho']
    vals = [v0, theta, kappa, xi, rho]
    ag = {}
    fn.start_recording()
    ids = []
    for i, p in enumerate(pnames):
        t = idouble(vals[i]); ag[p] = t.mark_as_input(); ids.append(t)
    cp = _proj_on_tape(ids[0], ids[1], ids[2], ids[3], ids[4], T, gp)
    res = cp.mark_as_output()
    fn.stop_recording()
    return fn, ag, res


def record_mc(S0, r, v0, kappa, theta, sigma, rho, T, K, n_steps):
    """Record Heston EulerLog MC on AADC tape."""
    dt = T/n_steps; n_z = 2*n_steps
    fn = aadc.Functions()
    ag = {}
    fn.start_recording()
    S_id = idouble(S0); ag['S0'] = S_id.mark_as_input()
    v0_id = idouble(v0); ag['v0'] = v0_id.mark_as_input()
    kappa_id = idouble(kappa); ag['kappa'] = kappa_id.mark_as_input()
    theta_id = idouble(theta); ag['theta'] = theta_id.mark_as_input()
    sigma_id = idouble(sigma); ag['sigma'] = sigma_id.mark_as_input()
    rho_id = idouble(rho); ag['rho'] = rho_id.mark_as_input()
    z_arr = aadc.array(np.random.randn(1, n_z))
    z_args = z_arr.mark_as_input_no_diff()
    rhohat = (idouble(1.0) - rho_id*rho_id)**0.5
    sdt = np.sqrt(dt)
    x = aadc.math.log(S_id); v = v0_id
    for t in range(n_steps):
        z_v = z_arr[0][t]*idouble(sdt)
        z_s = rho_id*z_arr[0][t]*idouble(sdt) + rhohat*z_arr[0][n_steps+t]*idouble(sdt)
        vplus = aadc.iif(v > 0, v, idouble(0.0))
        rtvplus = vplus**0.5
        x = x + (idouble(r) - vplus*idouble(0.5))*idouble(dt) + rtvplus*z_s
        sigma2 = sigma_id*sigma_id
        v = v + kappa_id*(theta_id-vplus)*idouble(dt) + sigma_id*rtvplus*z_v \
            + sigma2*(z_arr[0][t]*z_arr[0][t]*idouble(dt)-idouble(dt))*idouble(0.25)
    S_final = aadc.math.exp(x)
    payoff = aadc.iif(S_final > K, S_final - idouble(K), idouble(0.0))*idouble(np.exp(-r*T))
    res = payoff.mark_as_output()
    fn.stop_recording()
    return fn, ag, res, z_args, n_z


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    S0, r = 100.0, 0.05
    v0, kappa, theta, sigma, rho = 0.04, 2.0, 0.04, 0.3, -0.7
    T, K = 1.0, 100.0

    opt = Option_param(S0=S0, K=K, T=T, v0=v0, payoff='call')
    proc = Heston_process(mu=r, rho=rho, sigma=sigma, theta=theta, kappa=kappa)
    hp = Heston_pricer(opt, proc)

    # ── A. Fourier pricing ────────────────────────────────────────────────
    print("=" * 60)
    print("A. Fourier: AADC PROJ vs FMNM Fourier_inversion")
    print("=" * 60)

    ref = hp.Fourier_inversion()
    fn_f, ag_f, res_f = record_fourier(S0, K, T, r, v0, theta, kappa, sigma, rho)
    workers = aadc.ThreadPool(1)
    pnames_f = ['v0', 'theta', 'kappa', 'xi', 'rho']
    vals_f = [v0, theta, kappa, sigma, rho]
    inputs_f = {ag_f[pnames_f[j]]: vals_f[j] for j in range(5)}
    all_args_f = [ag_f[p] for p in pnames_f]

    result_f = aadc.evaluate(fn_f, {res_f: all_args_f}, inputs_f, workers)
    aadc_price = float(np.asarray(result_f[0][res_f]).flat[0])
    grads_f = {p: float(np.asarray(result_f[1][res_f][ag_f[p]]).flat[0]) for p in pnames_f}

    print(f"FMNM Fourier: {ref:.6f}")
    print(f"AADC PROJ:    {aadc_price:.6f}  (diff: {abs(aadc_price-ref):.2e})")

    # FD on FMNM
    h = 1e-6
    print(f"\nGradients (AADC vs FMNM FD):")
    fd_params = {'v0': v0, 'theta': theta, 'kappa': kappa, 'xi': sigma, 'rho': rho}
    proc_args = {'mu': r}
    for p in pnames_f:
        pu = fd_params.copy(); pu[p] += h
        pd = fd_params.copy(); pd[p] -= h
        opt_u = Option_param(S0=S0, K=K, T=T, v0=pu['v0'], payoff='call')
        opt_d = Option_param(S0=S0, K=K, T=T, v0=pd['v0'], payoff='call')
        proc_u = Heston_process(mu=r, rho=pu['rho'], sigma=pu['xi'], theta=pu['theta'], kappa=pu['kappa'])
        proc_d = Heston_process(mu=r, rho=pd['rho'], sigma=pd['xi'], theta=pd['theta'], kappa=pd['kappa'])
        fd = (Heston_pricer(opt_u, proc_u).Fourier_inversion()
            - Heston_pricer(opt_d, proc_d).Fourier_inversion()) / (2*h)
        ratio = grads_f[p]/fd if abs(fd) > 1e-12 else float('nan')
        print(f"  d/d({p:>5s}): AD={grads_f[p]:+10.6f}  FD={fd:+10.6f}  ratio={ratio:.4f}")

    # Benchmark
    n_iter = 500
    t0 = time.time()
    for _ in range(n_iter):
        aadc.evaluate(fn_f, {res_f: all_args_f}, inputs_f, workers)
    t_aadc_f = (time.time()-t0)/n_iter*1000

    t0 = time.time()
    for _ in range(n_iter):
        hp.Fourier_inversion()
    t_fmnm_f = (time.time()-t0)/n_iter*1000

    t0 = time.time()
    for _ in range(min(n_iter, 100)):
        hp.Fourier_inversion()
        for p in pnames_f:
            for s in [+1,-1]:
                pd = fd_params.copy(); pd[p] += s*h
                o = Option_param(S0=S0, K=K, T=T, v0=pd['v0'], payoff='call')
                pr = Heston_process(mu=r, rho=pd['rho'], sigma=pd['xi'], theta=pd['theta'], kappa=pd['kappa'])
                Heston_pricer(o, pr).Fourier_inversion()
    t_fmnm_fd = (time.time()-t0)/min(n_iter, 100)*1000

    print(f"\nBenchmark:")
    print(f"  AADC (price + 5 Greeks):  {t_aadc_f:.2f} ms")
    print(f"  FMNM Fourier (1 price):  {t_fmnm_f:.1f} ms")
    print(f"  FMNM + FD (11 evals):    {t_fmnm_fd:.0f} ms")
    print(f"  Speedup vs FD:           {t_fmnm_fd/t_aadc_f:.0f}x")

    # ── B. Calibration ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("B. Calibration: AADC Jacobian vs scipy FD")
    print("=" * 60)

    from scipy.optimize import least_squares

    ttms = [0.25, 0.5, 1.0, 2.0]
    strikes = np.array([90., 95., 100., 105., 110.])
    true_params = np.array(vals_f)  # [v0, theta, kappa, sigma, rho]
    n_instr = len(ttms) * len(strikes)

    # Market prices from FMNM
    market_prices = []
    for Ti in ttms:
        for Ki in strikes:
            o = Option_param(S0=S0, K=Ki, T=Ti, v0=v0, payoff='call')
            market_prices.append(Heston_pricer(o, proc).Fourier_inversion())
    market_prices = np.array(market_prices)
    print(f"Market: {len(ttms)} expiries x {len(strikes)} strikes = {n_instr} instruments")

    # Record tapes for all instruments
    tapes = [record_fourier(S0, Ki, Ti, r, v0, theta, kappa, sigma, rho)
             for Ti in ttms for Ki in strikes]

    def aadc_prices_jac(params):
        prices = np.zeros(n_instr)
        jac = np.zeros((n_instr, 5))
        for idx, (fn_i, ag_i, res_i) in enumerate(tapes):
            inp = {ag_i[pnames_f[j]]: params[j] for j in range(5)}
            aa = [ag_i[p] for p in pnames_f]
            r_ = aadc.evaluate(fn_i, {res_i: aa}, inp, workers)
            prices[idx] = float(np.asarray(r_[0][res_i]).flat[0])
            for j in range(5):
                jac[idx, j] = float(np.asarray(r_[1][res_i][aa[j]]).flat[0])
        return prices, jac

    def fmnm_prices(params):
        prices = []
        for Ti in ttms:
            for Ki in strikes:
                o = Option_param(S0=S0, K=Ki, T=Ti, v0=params[0], payoff='call')
                p = Heston_process(mu=r, rho=params[4], sigma=params[3],
                                   theta=params[1], kappa=params[2])
                prices.append(Heston_pricer(o, p).Fourier_inversion())
        return np.array(prices)

    x0 = np.array([0.06, 0.06, 1.0, 0.5, -0.3])
    bounds = ([1e-4, 1e-4, 0.01, 1e-4, -0.999], [1.0, 1.0, 50, 10, 0.999])

    n_runs = 3
    t0 = time.time()
    for _ in range(n_runs):
        r_aadc = least_squares(lambda p: aadc_prices_jac(p)[0] - market_prices,
                               x0=x0, jac=lambda p: aadc_prices_jac(p)[1],
                               bounds=bounds, method='trf')
    t_aadc_cal = (time.time()-t0)/n_runs

    t0 = time.time()
    for _ in range(n_runs):
        r_fd = least_squares(lambda p: fmnm_prices(p) - market_prices,
                             x0=x0, bounds=bounds, method='trf')
    t_fd_cal = (time.time()-t0)/n_runs

    print(f"\n  AADC:       {t_aadc_cal:.2f}s  (converged: {r_aadc.success})")
    print(f"  FMNM FD:    {t_fd_cal:.2f}s  (converged: {r_fd.success})")
    print(f"  Speedup:    {t_fd_cal/t_aadc_cal:.1f}x")
    print(f"  Max error:  {np.max(np.abs(r_aadc.x - true_params)):.2e}")

    # ── C. AITF ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("C. AITF: d(calibrated params) / d(market prices)")
    print("=" * 60)

    cal = r_aadc.x
    _, J = aadc_prices_jac(cal)
    dtheta_dP = np.linalg.solve(J.T @ J, J.T)

    atm_idx = ttms.index(1.0) * len(strikes) + list(strikes).index(100.)
    print(f"\n  Sensitivity to ATM 1Y (instrument #{atm_idx}):")
    for j in range(5):
        print(f"    d({pnames_f[j]:>5s}*)/dP = {dtheta_dP[j, atm_idx]:+.6f}")

    # FD verification
    h_m = 1e-4
    mu, md = market_prices.copy(), market_prices.copy()
    mu[atm_idx] += h_m; md[atm_idx] -= h_m
    ru = least_squares(lambda p: aadc_prices_jac(p)[0] - mu, x0=cal,
                       jac=lambda p: aadc_prices_jac(p)[1], bounds=bounds, method='trf')
    rd = least_squares(lambda p: aadc_prices_jac(p)[0] - md, x0=cal,
                       jac=lambda p: aadc_prices_jac(p)[1], bounds=bounds, method='trf')
    fd_s = (ru.x - rd.x) / (2*h_m)
    print(f"\n  FD verification:")
    for j in range(5):
        ratio = dtheta_dP[j, atm_idx] / fd_s[j] if abs(fd_s[j]) > 1e-12 else float('nan')
        print(f"    d({pnames_f[j]:>5s}*)/dP: IFT={dtheta_dP[j,atm_idx]:+.6f}  FD={fd_s[j]:+.6f}  ratio={ratio:.4f}")

    # AITF benchmark
    t0 = time.time()
    for _ in range(n_runs):
        _, Js = aadc_prices_jac(cal)
        np.linalg.solve(Js.T @ Js, Js.T)
    t_ift = (time.time()-t0)/n_runs

    t0 = time.time()
    for i in range(n_instr):
        mu2, md2 = market_prices.copy(), market_prices.copy()
        mu2[i] += h_m; md2[i] -= h_m
        least_squares(lambda p: aadc_prices_jac(p)[0] - mu2, x0=cal,
                      jac=lambda p: aadc_prices_jac(p)[1], bounds=bounds, method='trf')
        least_squares(lambda p: aadc_prices_jac(p)[0] - md2, x0=cal,
                      jac=lambda p: aadc_prices_jac(p)[1], bounds=bounds, method='trf')
    t_fd_ift = time.time() - t0

    print(f"\n  IFT:  {t_ift*1000:.0f} ms  (1 Jacobian + 5x5 solve)")
    print(f"  FD:   {t_fd_ift:.1f}s   ({n_instr}x2 recalibrations)")
    print(f"  Speedup: {t_fd_ift/t_ift:.0f}x")

    # ── D. Monte Carlo ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("D. Monte Carlo: AADC batch vs FMNM Cython MC")
    print("=" * 60)

    from FMNM.cython.heston import Heston_paths

    n_steps, n_paths = 50, 200000
    np.random.seed(42)
    all_Z = np.random.randn(n_paths, 2*n_steps)

    fn_mc, ag_mc, res_mc, z_args, n_z = record_mc(S0, r, v0, kappa, theta, sigma, rho, T, K, n_steps)
    pnames_mc = ['S0', 'v0', 'kappa', 'theta', 'sigma', 'rho']
    vals_mc = [S0, v0, kappa, theta, sigma, rho]
    inputs_mc = {ag_mc[pnames_mc[j]]: vals_mc[j] for j in range(6)}
    for zi in range(n_z):
        inputs_mc[z_args[0][zi]] = all_Z[:, zi].copy()

    workers4 = aadc.ThreadPool(4)
    all_args_mc = [ag_mc[p] for p in pnames_mc]

    t0 = time.time()
    result_mc = aadc.evaluate(fn_mc, {res_mc: all_args_mc}, inputs_mc, workers4)
    t_mc = time.time() - t0

    mc_price = np.average(result_mc[0][res_mc])
    mc_grads = {p: np.average(result_mc[1][res_mc][ag_mc[p]]) for p in pnames_mc}

    print(f"AADC MC ({n_paths} paths): {mc_price:.4f}  ({t_mc:.2f}s)")
    print(f"FMNM Fourier reference:  {ref:.4f}")

    print(f"\nGreeks:")
    for p in pnames_mc:
        print(f"  d/d({p:>6s}) = {mc_grads[p]:+.6f}")

    # FD check on delta (same paths)
    h_s = 1.0
    inp_up = inputs_mc.copy(); inp_up[ag_mc['S0']] = S0 + h_s
    inp_dn = inputs_mc.copy(); inp_dn[ag_mc['S0']] = S0 - h_s
    r_up = aadc.evaluate(fn_mc, {res_mc: []}, inp_up, workers4)
    r_dn = aadc.evaluate(fn_mc, {res_mc: []}, inp_dn, workers4)
    fd_delta = (np.average(r_up[0][res_mc]) - np.average(r_dn[0][res_mc])) / (2*h_s)
    print(f"\n  Delta: AD={mc_grads['S0']:.6f}  FD={fd_delta:.6f}  ratio={mc_grads['S0']/fd_delta:.4f}")

    # Honest MC benchmark: AADC vs FMNM Cython MC FD
    times_mc = []
    for _ in range(3):
        t0 = time.time()
        aadc.evaluate(fn_mc, {res_mc: all_args_mc}, inputs_mc, workers4)
        times_mc.append(time.time()-t0)
    t_mc_med = sorted(times_mc)[1]

    # FMNM Cython MC: 1 price + 6*2 FD bumps = 13 runs
    t0 = time.time()
    Heston_paths(N=n_steps, paths=n_paths, T=T, S0=S0, v0=v0, mu=r,
                 rho=rho, kappa=kappa, theta=theta, sigma=sigma)
    h_mc = {'S0': 1.0, 'v0': 0.005, 'kappa': 0.1, 'theta': 0.005, 'sigma': 0.03, 'rho': 0.01}
    mc_p = dict(S0=S0, v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    for pn in pnames_mc:
        for sign in [+1, -1]:
            pp = mc_p.copy(); pp[pn] += sign*h_mc[pn]
            Heston_paths(N=n_steps, paths=n_paths, T=T, S0=pp['S0'], v0=pp['v0'],
                         mu=r, rho=pp['rho'], kappa=pp['kappa'], theta=pp['theta'],
                         sigma=pp['sigma'])
    t_fmnm_mc_fd = time.time() - t0

    # Also measure single-threaded for fair comparison
    w1 = aadc.ThreadPool(1)
    times_mc1 = []
    for _ in range(3):
        t0 = time.time()
        aadc.evaluate(fn_mc, {res_mc: all_args_mc}, inputs_mc, w1)
        times_mc1.append(time.time()-t0)
    t_mc1 = sorted(times_mc1)[1]

    print(f"\n  AADC 4 threads (price + 6 Greeks): {t_mc_med:.2f}s  ({t_fmnm_mc_fd/t_mc_med:.0f}x vs FD)")
    print(f"  AADC 1 thread  (price + 6 Greeks): {t_mc1:.2f}s  ({t_fmnm_mc_fd/t_mc1:.0f}x vs FD)")
    print(f"  FMNM Cython MC + FD (13 runs):     {t_fmnm_mc_fd:.2f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary")
    print("=" * 60)
    print(f"  Fourier pricing:  AADC {t_aadc_f:.2f} ms vs FD {t_fmnm_fd:.0f} ms       ({t_fmnm_fd/t_aadc_f:.0f}x)")
    print(f"  Calibration:      AADC {t_aadc_cal:.2f}s vs FD {t_fd_cal:.2f}s       ({t_fd_cal/t_aadc_cal:.1f}x)")
    print(f"  AITF:             AADC {t_ift*1000:.0f} ms vs FD {t_fd_ift:.1f}s        ({t_fd_ift/t_ift:.0f}x)")
    print(f"  MC (4 threads):   AADC {t_mc_med:.2f}s vs FD {t_fmnm_mc_fd:.1f}s     ({t_fmnm_mc_fd/t_mc_med:.0f}x)")
    print(f"  MC (1 thread):    AADC {t_mc1:.2f}s vs FD {t_fmnm_mc_fd:.1f}s     ({t_fmnm_mc_fd/t_mc1:.0f}x)")
