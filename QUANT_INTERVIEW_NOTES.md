# Tier-1 Quant Developer Interview Knowledge Base

Topics 11–30. Dense, interview-oriented notes. Assumes calculus, probability, linear algebra, statistics, programming.

Notation: `S` = spot, `K` = strike, `T` = maturity, `τ = T − t` = time to expiry, `r` = continuously-compounded risk-free rate, `q` = continuous dividend yield, `σ` = volatility, `N(·)` = standard normal CDF, `n(·)` = standard normal PDF, `W_t` = Wiener process, `E`/`E^Q` = expectation under physical/risk-neutral measure.

---

# PART C — OPTIONS PRICING & GREEKS

## 11. Options Fundamentals

**Definitions.** An **option** is a contract conveying a *right, not an obligation*. A **call** gives the holder the right to *buy* the underlying at strike `K`; a **put** the right to *sell* at `K`. The writer (short) has the mirror obligation if assigned.

**Payoffs at expiry (European):**
- Call: `C_T = (S_T − K)^+ = max(S_T − K, 0)`
- Put: `P_T = (K − S_T)^+ = max(K − S_T, 0)`

**Long vs short.** Long option = paid premium, limited loss (premium), owns the right. Short option = received premium, obligation, potentially unlimited loss (short call is unbounded; short put loss capped at `K − 0` per unit because `S ≥ 0`).

**European vs American vs Bermudan.** European: exercise only at `T`. American: exercise any time in `[0,T]`. Bermudan: exercise on a discrete set of dates. American ≥ European in value always (superset of rights).

**Intrinsic vs time value.** Price `= intrinsic + time value`. Intrinsic call `= (S−K)^+`. Time value ≥ 0 for European options on non-dividend stock (proven via lower bound below); can be negative for a *European* deep-ITM put (its price can dip below intrinsic because you cannot access `K` until `T`). American options always have price ≥ intrinsic (exercise floor).

**Moneyness.** Call ITM if `S > K`, ATM `S ≈ K`, OTM `S < K` (reverse for puts). Quant refinements: **forward moneyness** `F/K` with `F = S·e^{(r−q)τ}`; **log-moneyness** `ln(F/K)`; **standardized moneyness** `ln(F/K)/(σ√τ)` (the argument that appears in `d1,d2`). "ATM" on a desk usually means ATM-forward (`K=F`), not `K=S`.

**No-arbitrage bounds (no model needed):**
- `0 ≤ C ≤ S` (a call ≤ owning the stock).
- Call lower bound: `C ≥ (S − K e^{−rτ})^+` (with dividends: `C ≥ (S e^{−qτ} − K e^{−rτ})^+`).
- Put bounds: `(K e^{−rτ} − S)^+ ≤ P ≤ K e^{−rτ}`.
- Proof of call lower bound: portfolio A = call + `K e^{−rτ}` cash; portfolio B = 1 share. A_T ≥ B_T in all states ⇒ A_0 ≥ B_0 ⇒ `C ≥ S − K e^{−rτ}`.

**Key theorem — never early-exercise an American call on a non-dividend stock.** Since `C_Eur ≥ S − K e^{−rτ} > S − K` (immediate-exercise value) for `r>0, τ>0`, the live option dominates exercising; selling beats exercising. Hence `C_Am = C_Eur`. Breaks with dividends (exercise optimal just before ex-div if dividend large enough). American *puts* can be optimal to exercise early even without dividends (payoff capped at `K`, so realizing `K−S` and earning interest can dominate).

**Synthetic positions (from put-call parity, Topic 12):** synthetic long stock = long call + short put (same K,T) + bond; synthetic call = long put + long stock; etc. Interviewers test that you can manufacture any leg from the others.

**Option strategies (decompose into legs; payoffs are linear in legs):**
- **Covered call**: long stock + short call → income, capped upside.
- **Protective put**: long stock + long put → insured downside (floor at K).
- **Collar**: long stock + long put(K1) + short call(K2) → bounded band, often zero-cost.
- **Bull call spread**: long call K1 + short call K2 (K1<K2) → capped bullish.
- **Bear put spread**: long put K2 + short put K1.
- **Straddle**: long call + long put same K → long volatility (V-shaped P&L), breakevens `K ± (c+p)`.
- **Strangle**: long call K2 + long put K1 (K1<K2) → cheaper long-vol, flat-bottom.
- **Butterfly**: long K1, short 2×K2, long K3 equally spaced → bet on low realized vol / pin at K2; a butterfly is a discrete second difference of payoff in K, and its price ≈ `e^{−rτ}·(risk-neutral density at K2)·(spacing)²` → Breeden–Litzenberger link to Topic 27.
- **Calendar spread**: same strike, different maturities → theta/vega play.
- **Risk reversal**: long call + short put → skew trade (quotes the vol skew in FX).

**Interview traps.** Payoff vs profit (profit subtracts premium, includes financing). Sign of short positions (mirror through x-axis). ×100 multiplier for equity options. Confusing `np.max` (reduction) with `np.maximum` (elementwise) in code. Assuming American call = European when there's a dividend.

**Real-firm use / quant dev.** Payoff and P&L engines are the base layer of any pricing library (C++ core, Python bindings via pybind11). Instruments → book → scenario engine that revalues vectorized under shocks. The kink at `S=K` is non-differentiable → breaks naive finite-difference Greeks and pathwise Monte-Carlo sensitivities near the strike (fix: payoff smoothing, likelihood-ratio method).

---

## 12. Put-Call Parity

**Statement (European, continuous dividend yield q):**
```
C − P = S e^{−qτ} − K e^{−rτ}
```
No dividends (`q=0`): `C − P = S − K e^{−rτ}`. Discrete dividends with PV `D`: `C − P = S − D − K e^{−rτ}`.

**Derivation (replication / arbitrage).** Portfolio A: long call + `K e^{−rτ}` cash. Portfolio B: long put + `e^{−qτ}` shares (reinvesting dividends). At `T`:
- A_T = `(S_T−K)^+ + K`.
- B_T = `(K−S_T)^+ + S_T`.
- Both equal `max(S_T, K)` in every state. Equal terminal value ⇒ equal price today (else arbitrage): `C + K e^{−rτ} = P + S e^{−qτ}`. ∎

**Arbitrage enforcement.** If `C − P > S e^{−qτ} − K e^{−rτ}` (call rich): sell call, buy put, buy `e^{−qτ}` shares, borrow the rest — locks riskless profit ("conversion"). Opposite mispricing → "reversal." This is *model-free*: holds regardless of the price process, only requires no-arbitrage, deterministic rates/dividends, no frictions, European exercise.

**Synthetic assets.** Rearrange: synthetic stock `= C − P + K e^{−rτ}`; synthetic call `= P + S e^{−qτ} − K e^{−rτ}`; synthetic bond `= P − C + S e^{−qτ}`. Used to trade the cheapest leg and to back out the *implied forward / implied dividend / implied borrow rate* from listed option prices (the put-call parity residual reveals hard-to-borrow / dividend expectations).

**Extensions.** American options give only inequalities: for non-dividend stock, `S − K ≤ C_Am − P_Am ≤ S − K e^{−rτ}`. FX options: `C − P = S e^{−r_f τ} − K e^{−r_d τ}` (foreign rate = dividend yield). Futures options (Black-76): `C − P = e^{−rτ}(F − K)`.

**Interview relevance.** The single most common no-arbitrage question. Also used to sanity-check implied vols: call and put IV at the same strike must match (parity implies same IV) — a mismatch signals stale quotes, dividend/borrow assumptions, or arb. Traps: forgetting dividends/carry, mixing spot vs forward, applying equality to American options.

---

## 13. Binomial Trees (Cox–Ross–Rubinstein)

**Why it exists.** A discrete, no-arbitrage pricing model that requires only replication; handles American exercise; converges to Black–Scholes. Pedagogically it *is* risk-neutral pricing made concrete.

**One-step model.** Stock `S₀ → uS₀` (prob real, irrelevant) or `dS₀`, with `d < e^{rΔt} < u` (no-arb). Replicate option paying `f_u, f_d` with `Δ` shares + `B` cash bond:
```
Δ = (f_u − f_d) / (S₀(u − d)),   B = e^{−rΔt}(u f_d − d f_u)/(u − d)
f₀ = ΔS₀ + B = e^{−rΔt}[ q f_u + (1−q) f_d ]
```
**Risk-neutral probability:**
```
q = (e^{(r−q_div)Δt} − d) / (u − d)
```
`q` is *not* the real-world probability; it is the measure under which discounted prices are martingales. Note `f₀` is independent of the real up-probability — this is the central insight.

**CRR parameterization.** `u = e^{σ√Δt}`, `d = 1/u = e^{−σ√Δt}`. Chosen so the tree recombines and matches the variance `σ²Δt` per step to first order. Alternatives: Jarrow–Rudd (equal real probabilities, `u,d = e^{(r−σ²/2)Δt ± σ√Δt}`), Tian, Leisen–Reimer (fast, smooth convergence).

**Multi-step & backward induction.** Recombining tree has `n+1` terminal nodes after `n` steps (O(n²) nodes). Compute terminal payoffs, then roll back: `f = e^{−rΔt}[q f_up + (1−q) f_down]` at each node.

**American options.** At each node take `max(continuation value, intrinsic value)`. This is dynamic programming for the optimal stopping problem. Trees are the standard textbook method for American options and give the early-exercise boundary.

**Convergence to Black–Scholes.** As `n→∞`, the binomial log-return is a sum of iid Bernoulli steps; CLT ⇒ log `S_T` → Normal `((r−σ²/2)τ, σ²τ)` under `q`, and the price → Black–Scholes. Convergence is `O(1/n)` but **oscillatory/sawtooth** (nodes straddle the strike differently as n changes). Fixes: average n and n+1 (or use even/odd), Leisen–Reimer (smooth, higher order), Richardson extrapolation, Broadie–Detemple smoothing at the layer adjacent to expiry, or Black–Scholes-smoothing the final step.

**Computational.** Time O(n²), memory O(n) if you roll a single 1-D array back. Vectorize each time slice with NumPy. Greeks: read Δ, Γ directly from the first two layers of the tree (no re-pricing) — a common efficiency question.

**Traps.** Using real-world probabilities instead of `q`. Forgetting the no-arb condition `d<e^{rΔt}<u`. Forgetting American check for puts/dividend calls. Non-recombining trees (O(2ⁿ)) when you accidentally make u·d ≠ d·u. Discrete dividends break recombination (use escrowed-dividend or a forward-shifted tree).

---

## 14. Black–Scholes(–Merton)

**Assumptions.** (1) Underlying follows GBM `dS = μS dt + σS dW` with constant `σ`; (2) constant `r`; (3) no arbitrage; (4) continuous frictionless trading, no transaction costs, infinitely divisible, short-selling allowed; (5) no dividends (or constant continuous yield `q`); (6) European exercise; (7) lognormal `S_T`.

**Derivation overview (two routes).**
1. **PDE / delta-hedging (Black–Scholes–Merton).** Form portfolio `Π = V − Δ S` with `Δ = ∂V/∂S`. Using Itô, the `dW` term cancels ⇒ `Π` is instantaneously riskless ⇒ must earn `r`. Yields the **Black–Scholes PDE**:
```
∂V/∂t + ½σ²S² ∂²V/∂S² + (r−q)S ∂V/∂S − rV = 0
```
with terminal condition `V(S,T) = payoff`. The equation is *preference-free* — `μ` vanished.
2. **Risk-neutral / Feynman–Kac.** Under `Q`, `dS = (r−q)S dt + σS dW^Q`; `V = e^{−rτ} E^Q[payoff]`. Evaluating the lognormal expectation gives the formula.

**Formula (European, dividend yield q):**
```
C = S e^{−qτ} N(d1) − K e^{−rτ} N(d2)
P = K e^{−rτ} N(−d2) − S e^{−qτ} N(−d1)
d1 = [ln(S/K) + (r − q + ½σ²)τ] / (σ√τ)
d2 = d1 − σ√τ
```
**Interpretation.** `N(d2)` = risk-neutral probability of finishing ITM (`Q(S_T>K)`). `S e^{−qτ}N(d1)` = PV of receiving the stock conditional on exercise (times the exercise prob under the stock-numéraire measure). `Δ_call = e^{−qτ}N(d1)`. In forward terms `d1,2 = [ln(F/K) ± ½σ²τ]/(σ√τ)`, `F = S e^{(r−q)τ}`.

**Inputs.** `S, K, τ, r, q` observable/deterministic; `σ` is the only unobservable → the market trades it as *implied volatility*. This is the crux: BS is used as a *quoting device* (a bijective map price↔vol), not as a literal model.

**Limit cases (sanity / interview favorites).**
- `σ→0`: `C → (S e^{−qτ} − K e^{−rτ})^+` (deterministic forward payoff).
- `τ→0`: `C → (S−K)^+` (intrinsic).
- `S→∞`: `C → S e^{−qτ} − K e^{−rτ}` (deep ITM, Δ→e^{−qτ}).
- `S→0`: `C→0`, `P→K e^{−rτ}`.
- Put-call parity recovered exactly from the formulas.
- ATM approximation: `C_ATM ≈ 0.4 · S · σ√τ` (from `C ≈ S σ√τ · n(0)`, `n(0)=0.399`) — must know this for mental math.

**Strengths.** Closed form, one intuitive free parameter, tractable Greeks, universal market language. **Weaknesses.** Constant vol (real markets show smile/skew, fat tails, jumps, stochastic vol); continuous costless hedging false; lognormal underestimates crash risk; single σ can't fit all strikes/maturities → motivates local vol, Heston, jumps.

**Traps.** `N(d2)` vs `N(d1)` meaning. Sign of `q` (and for FX, `q=r_f`). Forgetting `d1−d2=σ√τ`. Claiming BS "assumes risk neutrality" — it assumes no-arb; risk-neutrality is a *pricing technique*, real drift is `μ`. Vega and Gamma are identical for a call and put of same strike (parity: difference is linear in S, deterministic).

---

## 15. The Greeks

Sensitivities of `V` to inputs. Let `n(·)`=normal pdf. For a call (dividend yield q):

| Greek | Meaning | Call formula |
|---|---|---|
| **Delta** `∂V/∂S` | hedge ratio | `e^{−qτ}N(d1)` (put: `e^{−qτ}(N(d1)−1)`) |
| **Gamma** `∂²V/∂S²` | convexity, Δ-change | `e^{−qτ} n(d1) / (Sσ√τ)` (same call/put) |
| **Vega** `∂V/∂σ` | vol sensitivity | `S e^{−qτ} n(d1) √τ` (same call/put) |
| **Theta** `∂V/∂t` | time decay | see below |
| **Rho** `∂V/∂r` | rate sensitivity | `Kτ e^{−rτ}N(d2)` (put: `−Kτ e^{−rτ}N(−d2)`) |

**Theta (call):** `Θ = −e^{−qτ}S n(d1)σ/(2√τ) − rK e^{−rτ}N(d2) + qS e^{−qτ}N(d1)`. Usually negative for long options (time decay); can be positive for deep-ITM European puts.

**Properties & intuition.**
- Delta ∈ [0, e^{−qτ}] (call), [−e^{−qτ}, 0] (put); ≈ risk-neutral prob ITM; the number of shares to hedge.
- Gamma > 0 for long options, peaks ATM, explodes as `τ→0` ATM (pin risk); measures how often you must rebalance.
- Vega > 0 for long options, peaks ATM, grows with `√τ` — long-dated options are vega-heavy, short-dated are gamma-heavy.
- Theta and Gamma have opposite signs and are linked (below): you pay theta to own gamma.
- **The fundamental BS Greek identity** (from the PDE, q=0): `Θ + ½σ²S²Γ + rSΔ = rV`. For a delta-hedged portfolio (`Δ` netted), `Θ ≈ −½σ²S²Γ`: theta bleed is exactly the cost of positive gamma.

**Higher-order Greeks.** **Vanna** `∂²V/∂S∂σ` (delta's vol-sensitivity / skew hedging), **Volga/Vomma** `∂²V/∂σ²` (vega convexity, key for vol-of-vol / smile), **Charm** `∂Δ/∂t` (delta decay, overnight hedging), **Speed** `∂³V/∂S³`, **Color** `∂Γ/∂t`, **Veta** `∂Vega/∂t`. Vanna-Volga is a market pricing method for FX smiles.

**Practical meaning / firm use.** Desks manage the *book's aggregate Greeks*, not individual options. Delta hedged with underlying/futures; gamma and vega hedged with other options; risk reported as Greek ladders across spot/vol/time scenarios. Bucketed vega (per expiry/strike) because vol is not one number. "Long gamma" = profit from realized moves (buy low/sell high while rehedging); "short gamma" = market-maker's default risk that must be compensated by theta.

**Computational.** Analytic Greeks preferred (exact, cheap). Otherwise finite differences (central diff, common random numbers for MC), pathwise/likelihood-ratio methods, or automatic differentiation (adjoint AD / AAD is the production standard for computing thousands of Greeks in one backward pass — huge in XVA).

**Traps.** Gamma/Vega equal for call & put same strike. Vega is *not* a true Greek in a world of stochastic vol (BS assumes constant σ) — it's a sensitivity to a re-marked constant. Units: vega usually per 1 vol point (÷100), theta per calendar day (÷365), rho per 1% (÷100). Sign conventions for short books.

---

## 16. Delta Hedging & P&L

**Dynamic hedging.** Hold `−Δ_t` units of underlying against a long option, rebalancing as `Δ` changes. In continuous time with correct σ the hedge is perfect and replicates the option (BS derivation). The residual you're left holding is a pure volatility bet.

**Discrete-hedging P&L (the key result).** Over `[t, t+dt]`, a delta-hedged long-option P&L is, to second order:
```
dΠ ≈ ½ Γ (dS)² − ½ Γ (E[dS²]) = ½ S²Γ ( (dS/S)² − σ_impl² dt )
```
i.e. **P&L ≈ ½ Σ S²Γ (realized variance − implied variance)**. Being long gamma pays when realized vol > implied vol. This is the market maker's edge equation.

**Hedging error.** With `N` rebalances the hedging error has standard deviation `∝ 1/√N` (Boyle–Emanuel). Discrete rehedging leaves path-dependent P&L variance driven by gamma. Transaction costs grow with rehedge frequency ⇒ optimal frequency / no-trade band trade-off (Leland's model adjusts σ for costs; see notebook 4.1).

**Gamma effects.** Long gamma ⇒ delta auto-improves (you buy as price falls, sell as it rises → "buy low/sell high"), positive from moves but pays theta. Short gamma ⇒ you chase the market (sell low/buy high), collect theta, blow up in large moves. Pin risk near expiry ATM (gamma → ∞).

**Vega risk.** Delta-gamma hedging doesn't remove exposure to *changes in implied vol*. A book can be delta/gamma flat but lose on a vol repricing. Hedge vega with other options; but vega hedge is imperfect because of skew/term-structure (vanna, volga).

**P&L attribution (explain / "Greek P&L").** Decompose daily P&L via Taylor expansion:
```
ΔV ≈ Δ·ΔS + ½Γ·ΔS² + Vega·Δσ + Θ·Δt + Rho·Δr + ½Volga·Δσ² + Vanna·ΔS·Δσ + ...
```
**Unexplained P&L** = actual − predicted; large unexplained flags model error, stale marks, or missing risk (a core risk-management and interview topic). Regulatory "P&L attribution test" (FRTB) formalizes this.

**Firm use / quant dev.** Real-time Greek aggregation across the book; hedging engine computes target hedges and routes orders; P&L explain runs at EOD and intraday. Concerns: latency of revaluation, consistency of marks, handling of discrete dividends/earnings, corporate actions.

---

# PART D — STOCHASTIC CALCULUS

## 17. Brownian Motion (Wiener Process)

**From random walks.** Let `X_k = ±1` iid, `S_n = Σ X_k`. Rescale `W^{(n)}_t = S_{⌊nt⌋}/√n`. By Donsker's theorem this converges (in distribution, in path space) to standard Brownian motion `W_t`. BM is the continuous-time scaling limit of a symmetric random walk.

**Definition (standard BM `W_t`).**
1. `W_0 = 0`.
2. Independent increments.
3. `W_t − W_s ~ N(0, t−s)` for `s<t` (stationary Gaussian increments).
4. Continuous paths (a.s.).

**Properties.**
- `E[W_t]=0`, `Var(W_t)=t`, `Cov(W_s,W_t)=min(s,t)`.
- Gaussian process; Markov; martingale (`E[W_t | F_s]=W_s`).
- Paths are continuous but **nowhere differentiable** a.s.; of **infinite total variation** on any interval but **finite quadratic variation**.
- Self-similar (scaling): `W_{ct} =_d √c · W_t`. Time-inversion: `t W_{1/t}` is BM. Reflection principle gives running-max distribution.
- Law of iterated logarithm bounds oscillation.

**Quadratic variation (the crux).** For a partition of `[0,t]` with mesh → 0:
```
Σ (W_{t_{i+1}} − W_{t_i})² → t   (in L² / probability)
```
Written `[W]_t = t`, or the differential heuristic `(dW)² = dt`. This is *the* reason classical calculus fails: the second-order term does not vanish. Contrast: for a differentiable function quadratic variation is 0. Also `dW·dt = 0`, `(dt)² = 0`.

**Scaling / intuition.** Over `dt`, `dW ~ √dt` in magnitude — increments shrink like the square root of time, so squared increments accumulate linearly. This √t scaling underlies vol scaling (`σ_T = σ_1 √T`), the √n hedging error, and diffusion spreading.

**Geometric Brownian motion preview.** `S_t = S_0 exp((μ−½σ²)t + σW_t)` — always positive, lognormal, the BS underlying. Variants: arithmetic BM (can go negative — used for spreads/rates), Ornstein–Uhlenbeck (mean-reverting, notebook 6.1).

**Interview points.** Prove QV = t; explain why non-differentiability forces Itô; compute `E[W_t⁴]=3t²`, `E[W_s W_t]=min(s,t)`; know that `∫₀ᵗ W dW = ½W_t² − ½t` (the `−½t` is the Itô correction, not present in ordinary calculus).

---

## 18. Itô's Lemma

**Why.** For `f(W_t)`, a naive Taylor expansion `df = f'dW + ½f''(dW)²` has `(dW)² = dt` (not higher order), so the second-order term survives. Itô's lemma is the chain rule of stochastic calculus.

**Scalar form.** For `dX = a(X,t)dt + b(X,t)dW` and `f(X,t) ∈ C^{2,1}`:
```
df = ( ∂f/∂t + a ∂f/∂x + ½ b² ∂²f/∂x² ) dt + b ∂f/∂x dW
```
The extra `½ b² f_xx dt` is the **Itô correction**, arising from `(dX)² = b² dt`.

**Derivation.** Taylor to 2nd order: `df = f_t dt + f_x dX + ½ f_xx (dX)² + ...`. Substitute `dX`, use the multiplication table `(dt)²=0, dt·dW=0, (dW)²=dt`: `(dX)² = b²dt`. Collect. Rigorously it's the differential form of the Itô isometry / stochastic integral limit; the QV of X supplies the correction.

**Multiplication table.** `(dt)²=0`, `dt dW=0`, `dW_i dW_j = ρ_{ij} dt`.

**Multivariate.** For `X∈ℝⁿ`, `dX_i = a_i dt + Σ_k b_{ik} dW_k`, `f(X,t)`:
```
df = f_t dt + Σ_i f_{x_i} dX_i + ½ Σ_{i,j} f_{x_i x_j} (dX_i dX_j)
```
with `dX_i dX_j = (Σ_k b_{ik}b_{jk}) dt = (bbᵀ)_{ij} dt`. **Itô product rule:** `d(XY) = X dY + Y dX + dX dY` (the extra `dX dY` = covariation term vs ordinary calculus).

**Canonical examples.**
- `f=W²`: `d(W²) = 2W dW + dt` ⇒ `∫₀ᵗ W dW = ½W_t² − ½t`.
- **GBM:** `dS = μS dt + σS dW`, let `f=ln S`: `d ln S = (μ−½σ²)dt + σ dW` ⇒ `S_t = S_0 e^{(μ−½σ²)t + σW_t}`. The `−½σ²` is pure Itô correction and explains why `E[S_t]=S_0 e^{μt}` while the *median* grows at `μ−½σ²` (volatility drag).
- Derivation of BS PDE uses Itô on `V(S,t)`.

**Financial meaning.** Itô converts a model of the *underlying* into the dynamics of any *derivative* of it; the correction term is why convexity (gamma) generates P&L even with zero drift. Jensen/convexity ↔ the `½σ²S²Γ` term.

**Traps.** Forgetting the `½b²f_xx` term (the #1 error). Using Itô on non-C² payoffs (kinks/barriers → need Tanaka's formula / local time). Stratonovich vs Itô (Stratonovich obeys ordinary chain rule but isn't a martingale; finance uses Itô).

---

## 19. Stochastic Differential Equations & the Risk-Neutral Measure

**SDE.** `dX_t = a(X_t,t)dt + b(X_t,t)dW_t` interpreted as the integral eq `X_t = X_0 + ∫a ds + ∫b dW`. `a` = **drift** (deterministic pull), `b` = **diffusion** (random spread). Existence/uniqueness under Lipschitz + linear growth conditions.

**GBM (the workhorse).** `dS = μS dt + σS dW`. Solution via Itô on `ln S` (above): `S_t = S_0 exp((μ−½σ²)t + σW_t)`. Lognormal; `E[S_t]=S_0e^{μt}`, `Var=S_0²e^{2μt}(e^{σ²t}−1)`.

**Other key SDEs.**
- **Ornstein–Uhlenbeck / Vasicek:** `dX = κ(θ−X)dt + σ dW` — mean-reverting, Gaussian, can go negative; used for rates, spreads, stat-arb (notebooks 5.2, 6.1).
- **CIR:** `dX = κ(θ−X)dt + σ√X dW` — mean-reverting, non-negative if Feller `2κθ ≥ σ²`; variance process in Heston.
- **Arithmetic BM:** `dX = μdt + σdW`.

**Solving SDEs.** Techniques: (1) integrating factor for linear SDEs (OU); (2) Itô on a transform (GBM via ln); (3) recognize known form. Most SDEs have no closed form → discretize (Topic 22): **Euler–Maruyama** `X_{t+Δ}=X_t + a Δ + b √Δ Z`; **Milstein** adds `½ b b' (Δ² term)` i.e. `+ ½ b b'((√Δ Z)²−Δ)` for higher strong order.

**Risk-neutral measure `Q`.** Under physical `P`, `dS=μS dt+σS dW^P`. Pricing requires no-arb, which (First Fundamental Theorem of Asset Pricing) ⇔ existence of an equivalent measure `Q` under which the **discounted** asset `e^{−rt}S_t` is a martingale. Under `Q`:
```
dS = (r−q)S dt + σS dW^Q
```
Drift changes from `μ` to `r−q`; **diffusion σ is unchanged** (Girsanov only shifts drift). Prices are `V_t = e^{−r(T−t)} E^Q[ payoff | F_t ]`. Market completeness ⇔ `Q` unique (BS/binomial complete; stochastic-vol/jumps incomplete → many `Q`, need calibration).

**Interview points.** Derive GBM solution; explain why μ disappears from option prices (hedging removes it / change of measure); volatility drag `−½σ²`; difference between mean and median of `S_T`; Feller condition; why σ is measure-invariant.

---

## 20. Girsanov's Theorem & Martingales

**Martingale.** `M_t` adapted, integrable, with `E[M_t | F_s] = M_s` for `s≤t` — "no drift," best forecast of the future is the present. Sub/supermartingale for ≥/≤. Central because: (i) discounted tradables are `Q`-martingales; (ii) martingale representation underlies replication/hedging; (iii) optional stopping powers many results.

**Key martingales.** `W_t`; `W_t²−t`; **exponential (Doléans) martingale** `Z_t = exp(λW_t − ½λ²t)` (this being a martingale is exactly the MGF-normalization).

**Change of measure / Radon–Nikodym.** Two equivalent measures `P~Q` (same null sets). The **Radon–Nikodym derivative** `dQ/dP = Z_T` reweights probabilities; `E^Q[X] = E^P[X Z_T]`, and the density process `Z_t = E^P[Z_T|F_t]` is a `P`-martingale.

**Girsanov's theorem.** If `Z_t = exp(−∫₀ᵗ θ_s dW^P_s − ½∫₀ᵗ θ_s² ds)` is a true martingale (Novikov: `E[exp(½∫θ²ds)]<∞`), then under `Q` defined by `dQ/dP=Z_T`,
```
W^Q_t = W^P_t + ∫₀ᵗ θ_s ds   is a Q-Brownian motion.
```
Effect: **only the drift shifts by `−θ σ`-type terms; volatility/QV unchanged.** For GBM, choose the market price of risk `θ = (μ−r+q)/σ` (the **Sharpe-ratio-like risk premium**) so that the `Q`-drift becomes `r−q`.

**Equivalent Martingale Measure (EMM) & pricing.** FTAP1: no-arb ⇔ ∃ EMM. FTAP2: completeness ⇔ EMM unique. Risk-neutral pricing: `V_t/B_t = E^Q[V_T/B_T | F_t]` with numéraire `B_t=e^{rt}`. **Numéraire change**: any positive tradable can be a numéraire, inducing its own martingale measure (e.g. `T`-forward measure using the bond `P(t,T)` linearizes expectations of rates; stock measure gives the `N(d1)` term). Change-of-numéraire is how the two `N(d)` terms in BS get clean probabilistic meaning.

**Financial meaning.** Girsanov formalizes "switch off the risk premium." You are not asserting investors are risk-neutral; you are choosing a probability reweighting under which discounted prices have no drift, so expectations price by no-arb. The market price of risk `θ` is what you strip out.

**Interview points.** State Girsanov precisely (what changes, what doesn't); Novikov condition; derive the `Q`-drift of GBM; explain EMM/FTAP; numéraire change and the `T`-forward measure; why incomplete markets have non-unique `Q` (jumps, stoch vol).

---

## 21. Feynman–Kac

**What it is.** A bridge between (parabolic) PDEs and expectations of diffusions: the solution of a PDE = discounted expected value of a terminal payoff along an SDE, and vice versa.

**Statement.** If `u(x,t)` solves
```
∂u/∂t + a(x,t) ∂u/∂x + ½ b(x,t)² ∂²u/∂x² − r(x,t) u = 0,   u(x,T)=φ(x)
```
then
```
u(x,t) = E[ e^{−∫ₜᵀ r ds} φ(X_T) | X_t = x ],   dX = a dt + b dW.
```
(Add a source term `f` → adds `∫ e^{−∫r} f ds`.)

**Derivation (both directions).** Apply Itô to `Y_s = e^{−∫ₜˢ r} u(X_s,s)`. The `ds` drift is `e^{−∫r}(u_t + a u_x + ½b²u_xx − r u) = 0` by the PDE, so `Y` is a martingale (given the `dW` term is a true martingale). Then `u(x,t)=E[Y_t]=E[Y_T]=E[e^{−∫r}φ(X_T)]`. ∎ Conversely, define `u` by the expectation and show (Markov property + Itô) it satisfies the PDE.

**Black–Scholes relationship.** With `X=S`, `a=(r−q)S`, `b=σS`, discount `r`, terminal `φ=(S−K)^+`: Feynman–Kac says the BS PDE solution equals `e^{−rτ}E^Q[(S_T−K)^+]`. This *is* the equivalence of the PDE-hedging and risk-neutral-expectation routes to Black–Scholes. Evaluating the lognormal integral yields the closed form.

**Applications.** Justifies pricing exotics either by PDE (finite differences, Topic 23) or by simulation (Monte Carlo, Topic 22) — they solve the *same* problem. Extends to: bond pricing (`r` stochastic → term-structure PDEs), the Kolmogorov backward equation, and (with jumps) to PIDEs (notebooks 3.x). Multi-dimensional analog holds with `½ tr(bbᵀ D²u)`.

**Assumptions/limits.** Needs sufficient regularity (payoff, coefficients), the `dW` integral a true martingale (integrability), and typically a smooth or viscosity solution for non-smooth payoffs. Free-boundary problems (American options) become variational inequalities, not plain Feynman–Kac.

**Interview points.** State it; derive via the martingale argument; use it to explain "PDE ⇔ expectation"; connect to why Monte Carlo and finite differences are two numerical methods for one pricing problem; Kolmogorov backward/forward (Fokker–Planck) distinction.

---

# PART E — NUMERICAL METHODS

## 22. Monte Carlo Simulation

**Why.** Prices are risk-neutral expectations `V = e^{−rτ}E^Q[payoff]`. When no closed form exists (path-dependent, high-dimensional, exotic), estimate the expectation by simulating paths and averaging. Scales to high dimension where PDEs suffer the curse of dimensionality (MC error is dimension-independent).

**Path generation (GBM, exact).** Since `S_T = S_0 exp((r−q−½σ²)τ + σ√τ Z)`, `Z~N(0,1)`, simulate terminal `S_T` directly for European payoffs — no time-stepping needed, no discretization error. For path-dependent payoffs step exactly on a grid: `S_{t+Δ} = S_t exp((r−q−½σ²)Δ + σ√Δ Z)`. For SDEs without closed form: **Euler–Maruyama** (strong order ½, weak order 1) or **Milstein** (strong order 1).

**Estimator & error.** `V̂ = e^{−rτ} (1/M) Σ payoff(S^{(i)})`. Unbiased; by CLT the standard error is `σ_payoff/√M` → **convergence O(M^{−1/2})**, independent of dimension. To halve the error you need 4× the paths. Report a **confidence interval** (`±1.96·SE`) — interviewers always ask for the error bar.

**Variance reduction (multiply effective samples for same M):**
- **Antithetic variates.** Use `Z` and `−Z`. Averaging negatively-correlated pairs cuts variance if the payoff is monotone in `Z`. Cost: correlation must be negative to help; can backfire for non-monotone (e.g. straddle).
- **Control variates.** Use a correlated quantity with known expectation `Y` (e.g. the underlying itself, or a geometric-average Asian with closed form as control for arithmetic Asian). Estimator `X − β(Y − E[Y])`, optimal `β = Cov(X,Y)/Var(Y)`. Variance reduced by factor `1−ρ²`. The single most effective technique when a good control exists.
- **Importance sampling.** Change measure to sample the important region (deep-OTM options where payoff rarely nonzero). Reweight by likelihood ratio. Huge for rare events / tail risk; needs care to avoid blow-up of the ratio.
- **Stratified sampling / Latin hypercube.** Partition sample space to guarantee coverage.
- **Common random numbers.** Reuse the same `Z` across scenarios when computing Greeks by bumping → removes sampling noise from the difference.

**Quasi-Monte Carlo (QMC).** Replace pseudo-random with **low-discrepancy sequences** (Sobol, Halton). Error `O((log M)^d / M)` ≈ `O(1/M)` — much faster than `1/√M` in effective terms for moderate dimension. Needs Brownian bridge / PCA construction to front-load variance into the first (well-distributed) dimensions. No natural error estimate (deterministic) → use randomized QMC (scrambling) for confidence intervals. Production standard for many desks.

**Greeks in MC.** (1) **Bump-and-revalue** (finite difference) with common random numbers — simple, biased, noisy near kinks. (2) **Pathwise derivative** (differentiate payoff) — low variance, needs Lipschitz payoff (fails at digital/kink). (3) **Likelihood ratio** (differentiate the density) — works for discontinuous payoffs, higher variance. (4) **Adjoint AD (AAD)** — all Greeks in ~one backward pass, production standard.

**Complexity / engineering.** O(M·steps) time. Embarrassingly parallel → GPU/SIMD; use counter-based RNG (Philox) for reproducible parallel streams; avoid sharing a single Mersenne-Twister across threads. Vectorize with NumPy (draw the whole `(M×steps)` normal matrix). Memory: stream/aggregate rather than storing all paths; for path-dependent, store only needed running statistics. Watch RNG seeding for reproducibility and for correlated dimensions.

**Traps.** Reporting no confidence interval. Antithetics on non-monotone payoffs. Discretization bias vs statistical error (two different errors). Using MC for American options naively (needs Longstaff–Schwartz, Topic 26 — plain MC can't do the max over stopping times). Low-discrepancy sequences in high dimension without a good path construction.

---

## 23. Finite Difference Methods (for the Black–Scholes PDE)

**PDE.** `V_t + ½σ²S²V_{SS} + (r−q)S V_S − rV = 0`, terminal `V(S,T)=payoff`, plus boundary conditions in `S`. Solve backward in time on a grid. Often transform `x=ln S` to get constant coefficients: `V_t + (r−q−½σ²)V_x + ½σ²V_{xx} − rV = 0` (a convection–diffusion equation).

**Discretization.** Grid `(S_i, t_j)` or `(x_i,t_j)`. Approximate derivatives:
- `V_S ≈ (V_{i+1}−V_{i−1})/(2ΔS)` (central), `V_{SS} ≈ (V_{i+1}−2V_i+V_{i−1})/ΔS²`.
- Time derivative by forward/backward difference → three schemes.

**Explicit (forward in the backward march).** `V^{j} = A V^{j+1}` explicitly. Cheap per step but **conditionally stable**: needs `Δt ≤ ΔS²/(σ²S²_max)`-type CFL condition; equivalent to an *explicit trinomial tree*. Von Neumann stability requires the ratio `σ²Δt/Δx² ≤` const.

**Implicit (backward Euler).** Solve `A V^{j} = V^{j+1}` → a tridiagonal linear system each step (Thomas algorithm, O(N)). **Unconditionally stable**, first-order accurate in time `O(Δt)`, second order in space `O(ΔS²)`.

**Crank–Nicolson.** Average explicit + implicit → **second-order in time and space** `O(Δt²+ΔS²)`, unconditionally stable (A-stable). The standard choice. Caveat: only A-stable not L-stable → **spurious oscillations** for non-smooth payoffs (digitals, barriers, at the strike kink). Fixes: **Rannacher time-stepping** (start with 2 fully-implicit half-steps then CN), grid the strike/barrier on a node, or use a smoothed payoff.

**Consistency, stability, convergence (Lax equivalence theorem).** For a consistent scheme, **stability ⇔ convergence**. Consistency = local truncation error → 0. Stability = errors don't amplify (Von Neumann / matrix eigenvalue analysis). This triad is a guaranteed interview line.

**Boundary & grid.** Truncate `S∈[0,S_max]` (e.g. `S_max ≈ 3–5×K`); Dirichlet/Neumann/linearity (`V_{SS}=0`) boundary conditions. Non-uniform grids concentrate nodes near the strike/barrier for accuracy. American options → **operator splitting / PSOR (projected SOR)** or Brennan–Schwartz to impose `V ≥ payoff` (linear complementarity problem). PIDEs (jumps) add a nonlocal integral term (notebooks 3.x).

**Complexity.** Implicit/CN: O(N_S · N_t) with tridiagonal solves O(N_S) per step. 2-D (stochastic vol, two assets) → ADI (alternating direction implicit) to keep solves tractable; beyond 3-D, MC wins (curse of dimensionality).

**Traps.** CN oscillations on discontinuous payoffs (know Rannacher). Explicit-scheme stability limit. Putting the barrier/strike between nodes. Confusing consistency vs stability vs convergence. FD is best for low-dimension American/barrier; MC for high-dimension/path-dependent.

---

## 24. Implied Volatility

**Definition.** The `σ_impl` that makes the Black–Scholes price equal the observed market price: solve `BS(σ_impl) = V_market`. It is the market's forward-looking vol *quoted through the BS formula* — a change of variable, not a claim that BS is true.

**Existence/uniqueness.** BS price is continuous and **strictly increasing in σ** (vega `= S e^{−qτ}n(d1)√τ > 0`), from the no-arb lower bound (σ→0) up to the upper bound (σ→∞). So for an arbitrage-free price a unique `σ_impl>0` exists. This monotonicity is what makes root-finding well-posed.

**Smile / skew / term structure.** Plotting `σ_impl` vs strike is **not flat** (BS would predict flat): equity index shows a downward **skew** (OTM puts richer — crash fear, leverage effect); FX shows a symmetric **smile**; the shape varies with maturity → **term structure**; together → the **volatility surface** (Topic 27). Existence of smile ⇔ real returns are non-lognormal (fat tails, jumps, stochastic vol).

**Root-finding.**
- **Newton–Raphson:** `σ_{n+1} = σ_n − (BS(σ_n)−V_mkt)/vega(σ_n)`. Quadratic convergence; vega is the exact analytic derivative → very fast (usually <5 iters). Danger: vega → 0 for deep ITM/OTM or short expiry → division blows up / diverges.
- **Bisection/Brent:** bracket `[σ_low, σ_high]`, guaranteed convergence (linear/superlinear), robust — the safe production fallback. **Hybrid:** Newton with bisection safeguard (like `brentq`).
- **Good initial guess:** Brenner–Subrahmanyam ATM approx `σ_0 ≈ √(2π/τ)·(C/S)`; or Corrado–Miller. Jäckel's "Let's be rational" gives a near-machine-precision non-iterative implied vol (production standard).

**Numerical issues.** Vega vanishing (deep ITM/OTM, τ→0) → ill-conditioned; work in **total variance** `w=σ²τ` or in price-normalized units; enforce no-arbitrage bounds on the input price first (price outside `[intrinsic, S]` → no solution). Deep OTM quotes are noisy → weight by vega in calibration. Use put for OTM, call for the other side (parity) to keep away from tiny-vega regions.

**Calibration.** Fit a model (local vol, SVI, Heston) to the *surface* of implied vols by minimizing weighted squared error, subject to **no-arbitrage** (no calendar arbitrage: total variance increasing in T; no butterfly arbitrage: convexity in K). **SVI** (stochastic volatility inspired) parameterization of total variance per slice is the market standard for arbitrage-free interpolation. Vega-weighting and regularization stabilize the fit.

**Traps.** IV is not a forecast/realized vol; it's model-dependent (BS IV vs Bachelier/normal IV — rates markets quote **normal vol** since 2020 negative rates). Call and put IV must match by parity. Blindly Newton without safeguarding. Ignoring bid/ask → fit mid but respect the spread.

---

# PART F — ADVANCED MODELS & RISK

## 25. Heston Stochastic Volatility Model

**Motivation.** BS constant vol can't produce the smile/skew or vol clustering. Heston makes **variance itself stochastic and mean-reverting**, generating realistic smiles, and — crucially — retains a semi-closed-form via characteristic functions, so it calibrates fast.

**Dynamics (under Q).**
```
dS_t = (r−q) S_t dt + √v_t S_t dW_t^S
dv_t = κ(θ − v_t) dt + ξ √v_t dW_t^v
d⟨W^S, W^v⟩ = ρ dt
```
Parameters: `κ` mean-reversion speed, `θ` long-run variance, `ξ` vol-of-vol, `ρ` spot/vol correlation (typically negative for equities → skew), `v_0` initial variance.

**CIR variance process.** `v_t` follows a CIR/square-root process: mean-reverting, non-negative. **Feller condition** `2κθ ≥ ξ²` keeps `v_t>0` (else it can touch 0). Its transition density is noncentral chi-squared.

**Financial meaning of parameters.** `ρ<0` → left skew (spot down ↔ vol up, the leverage effect); `ξ` controls smile convexity (curvature/wings); `θ,v_0` set overall level and short/long term structure; `κ` sets how fast term structure flattens to `θ`.

**Characteristic function & pricing.** `S_t`'s log-price has a known **characteristic function** `φ(u)=E^Q[e^{iu ln S_T}]` (closed form, exponential-affine in `v_0`). Price European options via **Fourier inversion**: Heston's original formula (two `P_1,P_2` integrals), or **Carr–Madan FFT** (damped payoff transform, prices a whole strike grid in one FFT), or the **COS method** (Fourier-cosine expansion, fast and accurate). This is why Heston is practical despite no simple price formula. (Repo notebooks 1.3, 1.4.)

**Numerical care.** The classic characteristic function has a **branch-cut/discontinuity** in the complex log → use the **Little Heston Trap** (Albrecher et al.) formulation for numerical stability across maturities. Monte-Carlo simulation of CIR needs care (Euler can go negative): use **full truncation**, **QE scheme (Andersen)**, or exact sampling (Broadie–Kaya).

**Calibration.** Fit `(κ,θ,ξ,ρ,v_0)` to market implied-vol surface by least squares (vega-weighted), usually with FFT/COS pricing inside the loop; non-convex → use good initial guesses / global-then-local optimizers; watch identifiability (κ and ξ partly confounded). (Notebook 4.2.)

**Strengths.** Smile/skew + term structure with 5 interpretable params; semi-analytic → fast calibration; mean-reversion matches vol clustering. **Weaknesses.** Cannot fit very short-dated smiles (no jumps → too little short-term skew/kurtosis) → extend to **Bates** (Heston + jumps); single-factor vol misses complex term structure; Feller often violated in calibrated params; correlation constant.

**Interview points.** Write the SDEs; explain each parameter's smile effect; Feller condition; why the char function enables pricing (Fourier); Heston vs local vol (Dupire) vs jump models; the leverage effect via ρ.

---

## 26. American Options & Longstaff–Schwartz (LSM)

**Early exercise = optimal stopping.** American price `= sup_{τ∈[0,T]} E^Q[e^{−rτ} payoff(S_τ)]` over all stopping times `τ`. Equivalent to the dynamic-programming / Snell envelope: `V_t = max(payoff_t, e^{−rΔt}E^Q[V_{t+Δt}|F_t])` — at each instant, exercise iff intrinsic ≥ continuation value. The **exercise boundary** separates the two regions.

**Methods.** (1) **Binomial/trinomial trees** — natural, take max at each node (Topic 13). (2) **Finite differences / PSOR** — solve the LCP / variational inequality (Topic 23). (3) **Longstaff–Schwartz (LSM)** — regression-based Monte Carlo, the go-to for high-dimensional/path-dependent Americans (basket, Bermudan swaptions) where trees/PDE fail the curse of dimensionality.

**Why MC is hard for Americans.** Plain MC computes an expectation along fixed paths, but early exercise needs the **continuation value** (a conditional expectation) at each step to decide exercise — you can't see the future along a single path. LSM estimates that conditional expectation by cross-sectional regression.

**LSM algorithm.**
1. Simulate `M` paths of `S` forward to `T`.
2. Initialize cashflow = exercise payoff at `T`.
3. Step **backward** `t = T−Δ, …, Δ`. At each exercise date, among **in-the-money paths only**, regress the discounted realized future cashflow (the target) on basis functions of the current state (e.g. `1, S, S², Laguerre polynomials`) → fitted `Ĉ(S)` = estimated **continuation value**.
4. Exercise on paths where `intrinsic > Ĉ(S)`; set that path's cashflow to the intrinsic (and zero out later flows); else keep continuation.
5. Price = mean of discounted resulting cashflows over all paths.

**Why ITM-only & regression.** Regress only ITM paths (exercise decision only matters there → less noise, better fit). Basis functions approximate the conditional expectation `E[continuation | S_t]`; low-order polynomials usually suffice. Using the *regressed* value for the *decision* but the *realized* cashflow for the *value* reduces bias.

**Bias & convergence.** LSM gives a **low-biased** estimator (suboptimal exercise rule) — a valid lower bound. Regression error → the price is biased; converges as `M→∞` and basis richness →∞ (Clément–Lamberton–Protter proof). **Foreshadowing / dual methods** (Andersen–Broadie, Rogers) give an **upper bound** via martingale duality → sandwich the true price. Using in-sample paths for both regression and valuation adds foresight bias; use separate paths for a clean lower bound.

**Complexity/engineering.** O(M · steps · basis) plus regression (normal equations O(M·k²) or SVD for stability). Store paths or regenerate. Choice/number of basis functions is the main tuning knob; polynomial explosion in high dimension → use problem-specific features. Regularize regression to avoid overfitting the wings.

**Interview points.** Formulate as optimal stopping / Snell envelope; explain the continuation-value regression; why ITM-only; low-bias and the dual upper bound; when to use LSM vs tree vs PDE; American put early-exercise intuition (Topic 11).

---

## 27. Volatility Surface

**Object.** `σ_impl(K, T)` — BS implied vol as a function of strike and maturity. Equivalent coordinates: log-moneyness `k=ln(K/F)`, or **total implied variance** `w(k,T)=σ_impl²T` (the natural no-arbitrage variable). Built from liquid market option quotes.

**Smile & skew.** A cross-section (fixed T) is the **smile**; its slope is the **skew**. Equity indices: pronounced negative skew (OTM puts expensive) — crash-o-phobia + leverage effect (ρ<0). Single stocks: flatter/smile. FX: quoted as ATM vol, **25-delta risk reversal** (skew) and **25-delta butterfly** (smile convexity). Short-dated smiles are steepest (jump/kurtosis dominated); long-dated flatten toward ATM level.

**Term structure.** ATM vol vs T: typically upward-sloping in calm markets (mean-reversion toward higher long-run vol), inverted/backwardated in stress (front-end spikes). Encodes forward variance: **forward vol** between `T_1,T_2` from `σ²(T_2)T_2 − σ²(T_1)T_1`.

**Why it exists.** BS assumes one σ; markets price in fat tails, jumps, stochastic vol, supply/demand for protection → different vols per strike/maturity. The surface is the market's non-lognormal risk-neutral density made visible.

**Breeden–Litzenberger.** The risk-neutral density of `S_T` = `e^{rT} ∂²C/∂K²`. So the surface's curvature in strike *is* the implied distribution — connects directly to butterfly spreads (Topic 11) and to model calibration.

**Construction / no-arbitrage.** Interpolate/extrapolate quotes into a smooth arbitrage-free surface. Constraints: **no calendar-spread arbitrage** — total variance `w(k,T)` non-decreasing in T; **no butterfly arbitrage** — call price convex in K (density ≥ 0), plus slope bounds on `w` in k. **SVI** (Gatheral) `w(k)=a+b(ρ(k−m)+√((k−m)²+σ²))` is the industry-standard per-slice parameterization; **SSVI/eSSVI** enforce a globally arbitrage-free surface. Alternatives: local volatility (Dupire, exactly reprices the surface but has unrealistic dynamics), spline/kernel methods.

**Local vol (Dupire).** Unique deterministic `σ_loc(S,t)` reproducing all vanilla prices: `σ_loc² = (∂C/∂T + (r−q)K∂C/∂K + qC)/(½K²∂²C/∂K²)`. Fits today's surface perfectly but predicts a *flattening* forward smile (wrong dynamics) → SLV (stochastic-local vol) blends Dupire with Heston-type dynamics for exotics.

**Firm use / quant dev.** The surface is a core market-data object rebuilt continuously from the options feed; must be arbitrage-free, smooth, fast to query and to bump for Greeks; serves pricing of exotics and risk of the vanilla book. Interpolation latency and arbitrage-repair are real engineering problems.

**Interview points.** Explain skew vs smile and their drivers; total variance & no-arb conditions; Breeden–Litzenberger; SVI; local vol vs stochastic vol dynamics ("sticky strike" vs "sticky delta/moneyness" regimes and their delta implications).

---

## 28. VaR & CVaR (Expected Shortfall)

**VaR.** Value-at-Risk at confidence `α` over horizon `h`: the loss `L` such that `P(L > VaR_α) = 1−α`. Formally the α-quantile of the loss distribution: `VaR_α = inf{ℓ : P(L≤ℓ) ≥ α}`. "With 99% confidence, 1-day loss ≤ VaR." A quantile, not an average.

**Three computation methods.**
1. **Parametric (variance–covariance / delta-normal).** Assume returns `~N(μ,Σ)`. Portfolio σ_P = √(wᵀΣw); `VaR_α = −(μ_P − z_α σ_P)` where `z_α=Φ^{−1}(α)` (e.g. 2.326 at 99%). Fast, closed-form; **fails for fat tails and nonlinear (option) portfolios**. Delta-gamma extension adds convexity (Cornish–Fisher for skew/kurtosis).
2. **Historical simulation.** Reprice/aggregate the portfolio under the last `N` actual return scenarios; VaR = empirical quantile of the P&L vector. No distributional assumption, captures fat tails/correlations as realized; but limited by sample, assumes the past = future, slow to react, ghost effects when scenarios drop out.
3. **Monte Carlo.** Simulate risk factors from a model, full-revalue the portfolio, take the quantile. Handles nonlinearity/path dependence; expensive (nested simulation for options → use grids/AAD/proxy).

**CVaR / Expected Shortfall.** `ES_α = E[L | L ≥ VaR_α]` — the average loss in the worst `1−α` tail. `ES_α = (1/(1−α)) ∫_α^1 VaR_u du`. Answers "how bad when it's bad," which VaR ignores.

**Coherence (Artzner et al.).** A coherent risk measure is monotone, translation-invariant, positively homogeneous, and **subadditive** (`ρ(A+B) ≤ ρ(A)+ρ(B)` — diversification never increases risk). **VaR is not subadditive** (can penalize diversification, esp. fat tails/discrete payoffs) → not coherent. **ES is coherent** → theoretically preferred and now favored by regulators.

**Regulatory usage.** Basel II/2.5 used 99% 10-day VaR; **FRTB (Basel III)** replaces VaR with **97.5% Expected Shortfall**, with liquidity-horizon scaling and the P&L attribution test (ties back to Topic 16). Backtesting: **Kupiec** (unconditional coverage) and **Christoffersen** (independence of exceptions) tests on VaR breaches; ES is harder to backtest (not elicitable alone) → Acerbi–Székely tests.

**Weaknesses / traps.** VaR says nothing about tail beyond the quantile; non-subadditive; procyclical; model/estimation risk; horizon scaling (√t only valid iid normal); overlapping vs non-overlapping windows; confusing 1-day vs 10-day, and `α` conventions (95 vs 99). Stress testing / scenario analysis complements VaR (forward-looking, non-statistical). CVaR is convex → nicer for optimization (Rockafellar–Uryasev LP formulation).

**Interview points.** Define precisely (quantile); the three methods with pros/cons; why ES over VaR (subadditivity counterexample); FRTB 97.5% ES; backtesting; scaling assumptions; delta-gamma VaR for option books.

---

## 29. Factor Models

**Idea.** Explain asset returns by a few common **factors** plus idiosyncratic noise: `r_i = α_i + Σ_k β_{ik} f_k + ε_i`. Reduces the covariance matrix from O(n²) free params to factor structure `Σ = BΩBᵀ + D` (B loadings, Ω factor cov, D diagonal idio) → estimable, invertible, the backbone of risk models and portfolio construction.

**CAPM.** Equilibrium single-factor: `E[r_i] − r_f = β_i (E[r_m] − r_f)`, `β_i = Cov(r_i,r_m)/Var(r_m)`. Only **systematic risk** (β) is priced; idiosyncratic risk is diversifiable → no premium. Derived from mean-variance efficiency (everyone holds the market + risk-free → the market is the tangency portfolio). SML plots `E[r]` vs β. **Assumptions:** homogeneous expectations, one-period, no frictions, mean-variance investors. **Empirical failures:** low-beta anomaly, size, value, momentum → multi-factor.

**Fama–French 3-factor.** `r_i − r_f = α_i + β_mkt·MKT + β_smb·SMB + β_hml·HML + ε`. **SMB** (small minus big, size), **HML** (high minus low book-to-market, value). Carhart adds **UMD/MOM** (momentum) → 4-factor; FF **5-factor** adds **RMW** (profitability) and **CMA** (investment). Factors are long-short mimicking portfolios.

**Types of factor models.** (1) **Macroeconomic** (observable factors: GDP, inflation, rates — Chen–Roll–Ross). (2) **Fundamental** (observable characteristics as loadings: BARRA/Axioma industry+style factors; cross-sectional regression each period). (3) **Statistical** (PCA/factor analysis: latent factors from the return covariance — no economic label). Know which is which.

**Estimation.** Time-series regression (factors observed → estimate β) vs cross-sectional (loadings observed → estimate factor returns each period, Fama–MacBeth two-pass with the standard-error correction). PCA for statistical factors. Shrinkage (Ledoit–Wolf) on the covariance.

**Risk attribution / portfolio applications.** Decompose portfolio variance into **factor (systematic) risk** `βᵀΩβ` + **specific risk** `wᵀDw`; compute **marginal contribution to risk** `∂σ_P/∂w_i`, factor exposures, and active risk (tracking error) vs a benchmark. Used for: risk budgeting, hedging unwanted factor exposure, performance attribution (α vs factor β), and constrained mean-variance/Black–Litterman optimization (repo notebook 7.1). Alpha = return unexplained by factors (the thing quant researchers hunt).

**Interview points.** Derive CAPM β; distinguish systematic vs idiosyncratic; the three model types; APT vs CAPM (APT = no-arb multifactor, no equilibrium/market-portfolio assumption); why factor structure makes Σ invertible/stable; Fama–MacBeth; factor zoo / multiple-testing critique; long-short construction of factors.

---

# PART G — MARKET MICROSTRUCTURE

## 30. Market Microstructure

**Why it matters for a quant dev.** This is where theory meets the wire. HFT/market-making firms (HRT, Jump, Optiver, IMC, Citadel Securities) live and die on understanding exchange mechanics, latency, and adverse selection. Interviews test both the *economics* (spread, impact, adverse selection) and the *systems* (matching engines, protocols, latency).

**Exchanges & the order book.** A modern equity/derivatives exchange runs a **central limit order book (CLOB)** per instrument: resting **limit orders** on the bid (buyers) and ask (sellers) sides, sorted by price then time. **Best bid/offer (BBO)** = top of book; **NBBO** = national best across US venues (Reg NMS). Depth = quantity at each level.

**Matching engine & priority.** The engine matches incoming orders against resting liquidity. Dominant rule: **price–time priority (FIFO)** — best price first, then earliest arrival. Alternatives: **pro-rata** (allocate proportionally to size, common in some futures/rates — changes optimal order sizing), or **pro-rata with a top-order/priority allotment**. Engines are single-threaded per symbol for deterministic sequencing, process a serialized event stream, and publish market-data updates. Determinism and low, predictable latency are the design goals.

**Order types.**
- **Market order** — immediate execution at best available; pays the spread, guaranteed fill, price uncertain; consumes liquidity (aggressive/taker).
- **Limit order** — execute only at price ≤/≥ limit; provides liquidity (passive/maker), risk of non-execution and of **adverse selection**.
- **Stop / stop-limit** — dormant until trigger price, then becomes market/limit; used for risk exits; can cascade (flash-crash amplifier).
- **IOC / FOK** (immediate-or-cancel / fill-or-kill), **GTC/day**, **peg** orders.
- **Iceberg / reserve** — displays only a small tip, hides the rest; reduces information leakage but loses time priority on the hidden portion when it refreshes.
- **Hidden / midpoint** orders and **dark pool** liquidity (non-displayed, execute at midpoint, reduce impact for large orders).

**Bid–ask spread & its decomposition.** Spread compensates the liquidity provider for three costs (Stoll / Glosten–Milgrom / Huang–Stoll): (1) **order-processing** (fixed/technology), (2) **inventory** (holding risk of unwanted position), (3) **adverse selection** (trading against better-informed counterparties). Effective spread `= 2·|trade price − mid|`; realized spread strips out the post-trade mid move (the market-maker's actual capture); the difference is the adverse-selection cost.

**Adverse selection (the central concept).** A resting limit order gets filled precisely when someone wants the other side — often because they *know* something (informed flow). So passive fills are biased toward being *wrong* right after (price moves against you). Glosten–Milgrom: the spread arises endogenously because market makers set bid/ask so expected loss to informed traders = expected gain from uninformed (noise) traders. **PIN** (probability of informed trading) and order-flow **toxicity** (VPIN) quantify it. A market maker's entire edge is managing adverse selection and inventory.

**Market impact.** Trading moves the price against you. **Temporary impact** (liquidity consumption, mean-reverts) vs **permanent impact** (information, persists). Empirically impact is **concave in size** — the "**square-root law**": `ΔP ≈ Y·σ·√(Q/V)` (impact ∝ volatility × √(participation)). Almgren–Chriss models the temporary/permanent split for optimal execution. Kyle's **λ** (from Kyle 1985) = price impact per unit order flow = `1/market depth`; Kyle's lambda is *the* microstructure measure of illiquidity.

**Price discovery & order flow.** Prices incorporate information via trading. **Order flow imbalance (OFI)** and signed volume are the strongest short-horizon predictors of price. **Trade classification** (Lee–Ready, tick rule) signs trades as buyer/seller-initiated. Roll's model infers the effective spread from the negative serial covariance of price changes (bid-ask bounce). Hasbrouck information share attributes price discovery across venues.

**FIX protocol & connectivity.** **FIX** (Financial Information eXchange) = the standard tag-value session/application messaging for order entry and drop-copies (NewOrderSingle=35=D, ExecutionReport=35=8, cancel/replace). Latency-sensitive venues use **binary native protocols** (e.g. ITCH for market data, OUCH for order entry; exchange-specific binary). Market data feeds: **direct feeds** (raw, per-exchange, e.g. ITCH) vs **consolidated SIP** (slower); firms build the book from incremental updates (add/modify/delete messages) — the **book builder** is a core low-latency component.

**Low-latency engineering (what a quant dev builds).**
- **Colocation** (servers in the exchange data center), cross-connects, equal-length cabling; latency measured in **nanoseconds**.
- **Kernel bypass** (Solarflare/Onload, DPDK), **busy-polling** (no interrupts), **NUMA pinning**, cache-line awareness, **lock-free** ring buffers, avoiding syscalls/allocations on the hot path.
- **FPGA/ASIC** for tick-to-trade in the fastest shops (parse feed, update book, fire order in hardware).
- **Microwave/laser** links for inter-city latency arbitrage (Chicago–NJ).
- Deterministic, jitter-free processing; **hardware timestamping** (PTP) for accurate latency measurement; extensive **replay/simulation** against recorded market data for testing.
- Data structures: the **limit order book** as arrays/intrusive lists indexed by price level (price-level → FIFO queue of orders), hash map order-id → node for O(1) cancel; contiguous memory, no pointer chasing.

**Market making.** Continuously quote two-sided; profit = spread capture − adverse selection − inventory risk. **Avellaneda–Stoikov** optimal MM: set bid/ask around a **reservation price** skewed by inventory (`r = s − q·γσ²(T−t)`) and a spread widening with `γσ²` and order-flow intensity `λ`. Skew quotes to mean-revert inventory to zero; widen in volatility; manage aggregate Greeks if quoting options.

**Execution algorithms (agency/optimal execution).**
- **TWAP** — slice evenly over time; ignores volume; simple, predictable, gameable.
- **VWAP** — match the volume-weighted average price by trading in proportion to the historical/expected intraday volume profile (U-shaped); benchmark for agency execution.
- **POV / participation** — trade a fixed % of market volume.
- **Implementation Shortfall (Almgren–Chriss)** — minimize `E[cost] + λ·Var[cost]`, trading off **market impact** (trade slow) vs **timing/volatility risk** (trade fast); yields an optimal front-loaded trajectory depending on risk aversion. IS = difference between decision price and final executed price (Perold) = the true cost benchmark.
- **Smart Order Routing (SOR)** — split a parent order across venues/dark pools to capture best price and liquidity under Reg NMS, minimizing fees (maker–taker rebates) and information leakage.

**Regulation to know.** Reg NMS (order protection/NBBO), MiFID II (Europe, best execution/transparency), tick-size regimes, maker–taker fees, circuit breakers/LULD.

**Interview points.** Explain adverse selection and why it creates the spread (Glosten–Milgrom); price–time vs pro-rata and how it changes order-placement strategy; square-root impact law and Kyle's λ; how a matching engine sequences and why single-threaded/deterministic; how you'd build/maintain an order book (data structures, O(1) cancel); TWAP vs VWAP vs IS; sources of latency and how to cut them (colo, kernel bypass, FPGA); what drives short-horizon price prediction (OFI). Systems interviews: design a low-latency order book; measure tick-to-trade; handle out-of-order/gapped market data.

---

## Cross-Topic Concept Map (how it all connects)

- **No-arbitrage** is the spine: bounds (11) → put-call parity (12) → replication in trees (13) → BS PDE & risk-neutral pricing (14, 19–21).
- **Two routes to a price, one answer:** PDE/hedging (14, 23) ⇔ risk-neutral expectation (14, 19–22), joined by **Feynman–Kac** (21) and **Girsanov** (20). Solve numerically by **Monte Carlo** (22, high-dim/path-dependent) or **finite differences** (23, low-dim/American).
- **Itô + QV** (17–18) is the engine that makes convexity (Γ) generate P&L → delta-hedging P&L = realized−implied variance (16) → why vol is *the* traded quantity → **implied vol / surface** (24, 27) → models that fit it (**Heston** 25, local vol).
- **American options** (26) = optimal stopping, solved by trees (13), PDE/PSOR (23), or LSM regression-MC (26).
- **Risk** (28) reuses MC/revaluation (22) and Greeks (15–16); **factor models** (29) supply the covariance for parametric VaR and portfolio construction.
- **Microstructure** (30) is where all prices/hedges are actually executed — impact and adverse selection are the real-world frictions the earlier idealized models omit.

## Highest-yield things to have instant recall of
- Payoffs, put-call parity (with dividends), and the "no early exercise of American call w/o dividends" argument.
- Risk-neutral pricing statement `V=e^{−rτ}E^Q[payoff]`; why μ→r.
- `(dW)²=dt`; Itô's lemma; GBM solution and volatility drag `−½σ²`.
- BS formula, meaning of `N(d1), N(d2)`, ATM approx `0.4·Sσ√τ`.
- Greek signs/shapes; `Θ ≈ −½σ²S²Γ`; delta-hedge P&L = ½S²Γ(realized−implied var).
- Girsanov (drift shifts, vol invariant) + Feynman–Kac (PDE↔expectation).
- MC error `O(1/√M)` + variance reduction; CN scheme + Lax (stability⇔convergence).
- VaR vs ES, subadditivity, FRTB 97.5% ES.
- CAPM β; systematic vs idiosyncratic; factor covariance structure.
- Adverse selection → spread; square-root impact law; price–time priority; TWAP/VWAP/IS; colo + kernel bypass + FPGA.
