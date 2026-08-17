"""Verification for the 2026-07-12 round (NG's capital-letter reply to AA's email).

Backs the new Remark "Joint Distribution of Attention and Beliefs" (rem:joint)
in model-2.tex and the E7 spec in exhibit_specs_v2.md.

Claim 1 (fixed rho, mu): with constant s and an interior optimum,
    t^TG = 1/2 + [rho*r - sigma*(1+r)*(1-s)]/mu   is BILINEAR in (sigma, s),
    so  E[t] - E_indep[t] = (1+r)/mu * Cov(sigma, s)  EXACTLY.
Claim 1b (heterogeneous ratios, the version stated in rem:joint): with
    attention ratios (rho/mu, sigma/mu) jointly distributed with beliefs s,
    and independence defined as attention-block independent of beliefs with
    marginals fixed,  E[t] - E_indep[t] = (1+r) * Cov(sigma/mu, s)  EXACTLY
    (rho/mu enters linearly, so no other covariance moves the mean).
Claim 2: the UG offer t^UG = 1/2 + (b*rho - sigma*(a+b))/(mu + 2*b*sigma)
    is NOT bilinear in attention and beliefs (b in the denominator), so the
    UG independence gap has no covariance formula -> computed numerically in E7.
Claim 3: under the declining schedule of rem:return,
    t^TG = [mu/2 + rho*r - sigma*(1+r)*(1-s0)] / [mu + 2*sigma*(1+r)*s1],
    bilinearity fails for s1 > 0; the covariance formula is exact only in the
    constant-s limit s1 = 0 (which the expression nests).

Run:  python3 ng_reply_round2_checks.py   ->   "FAILURES: none" expected.
"""
import sympy as sp

rho, sig, mu, r, s, a, b = sp.symbols('rho sigma mu r s a b', positive=True)

fails = []

# ---------- Claim 1: exact covariance formula (constant-s TG, fixed rho, mu) ----------
tTG = sp.Rational(1, 2) + (rho*r - sig*(1+r)*(1-s))/mu

# bilinearity: second own-derivatives vanish, cross-derivative is the constant (1+r)/mu
if sp.simplify(sp.diff(tTG, sig, 2)) != 0 or sp.simplify(sp.diff(tTG, s, 2)) != 0 \
        or sp.simplify(sp.diff(tTG, sig, s) - (1+r)/mu) != 0:
    fails.append("Claim 1 bilinearity")

# exact gap on an arbitrary 2x2 joint distribution of (sigma, s)
p11, p10, p01 = sp.symbols('p11 p10 p01', positive=True)
p00 = 1 - p11 - p10 - p01
sL, sH, gL, gH = sp.symbols('s_L s_H g_L g_H', positive=True)  # g = sigma
joint = {(gL, sL): p00, (gL, sH): p01, (gH, sL): p10, (gH, sH): p11}
Et = sum(pr * tTG.subs({sig: g_, s: s_}) for (g_, s_), pr in joint.items())
Eg = sum(pr * g_ for (g_, s_), pr in joint.items())
Es = sum(pr * s_ for (g_, s_), pr in joint.items())
Egs = sum(pr * g_ * s_ for (g_, s_), pr in joint.items())
cov = sp.expand(Egs - Eg*Es)
Et_ind = sp.expand(
    (p00+p01)*(p00+p10)*tTG.subs({sig: gL, s: sL}) +
    (p00+p01)*(p01+p11)*tTG.subs({sig: gL, s: sH}) +
    (p10+p11)*(p00+p10)*tTG.subs({sig: gH, s: sL}) +
    (p10+p11)*(p01+p11)*tTG.subs({sig: gH, s: sH})
)
gap = sp.simplify(sp.expand(Et) - Et_ind - (1+r)/mu * cov)
if gap != 0:
    fails.append(f"Claim 1 exact gap: residual {gap}")

# ---------- Claim 1b: heterogeneous ratios, attention block vs beliefs ----------
# t = 1/2 + (rho/mu)*r - (sigma/mu)*(1+r)*(1-s); two attention types (rt_i, st_i),
# two belief values s_j, arbitrary joint q_ij; independence = product of the
# attention-block and belief marginals. Note rho/mu VARIES with the attention type:
# its covariances drop out because it enters linearly and its marginal is preserved.
rt1, rt2, st1, st2 = sp.symbols('rt1 rt2 st1 st2', positive=True)
q11, q12, q21 = sp.symbols('q11 q12 q21', positive=True)
q22 = 1 - q11 - q12 - q21


def tfun(rt_, st_, s_):
    return sp.Rational(1, 2) + rt_*r - st_*(1+r)*(1-s_)


cells = [((rt1, st1), sL, q11), ((rt1, st1), sH, q12),
         ((rt2, st2), sL, q21), ((rt2, st2), sH, q22)]
Et2 = sum(q_*tfun(A[0], A[1], s_) for A, s_, q_ in cells)
rowm = {(rt1, st1): q11+q12, (rt2, st2): q21+q22}
colm = {sL: q11+q21, sH: q12+q22}
Et2_ind = sum(rowm[A]*colm[s_]*tfun(A[0], A[1], s_) for A in rowm for s_ in colm)
Est = (q11+q12)*st1 + (q21+q22)*st2
Es2 = colm[sL]*sL + colm[sH]*sH
Ests = q11*st1*sL + q12*st1*sH + q21*st2*sL + q22*st2*sH
cov2 = sp.expand(Ests - Est*Es2)
gap2 = sp.simplify(sp.expand(Et2) - sp.expand(Et2_ind) - (1+r)*cov2)
if gap2 != 0:
    fails.append(f"Claim 1b exact gap (ratio version): residual {gap2}")

# ---------- Claim 2: UG optimum not bilinear ----------
tUG = sp.Rational(1, 2) + (b*rho - sig*(a+b))/(mu + 2*b*sig)
# sanity: matches known comparative statics from prop:ug
if sp.simplify(sp.diff(tUG, rho) - b/(2*b*sig + mu)) != 0:
    fails.append("UG sanity: d t/d rho != b/(2b sigma + mu)")
if sp.simplify(sp.diff(tUG, sig, 2)) == 0:
    fails.append("Claim 2: t^UG unexpectedly bilinear in sigma")

# ---------- Claim 3: endogenous-s TG not bilinear in (sigma, s0) for s1 > 0 ----------
s0, s1 = sp.symbols('s0 s1', positive=True)
tTG_end = (mu/2 + rho*r - sig*(1+r)*(1-s0)) / (mu + 2*sig*(1+r)*s1)
if sp.simplify(sp.diff(tTG_end, sig, 2)) == 0:
    fails.append("Claim 3: endogenous-s t^TG unexpectedly bilinear")
if sp.simplify(sp.diff(tTG_end.subs(s1, 0), sig, 2)) != 0:
    fails.append("Claim 3 nesting at s1=0 broken")

print("FAILURES:", fails if fails else "none — all claims verified")
