import sympy as sp

ok = lambda name, cond: print(("  PASS  " if cond else "  FAIL  ") + name)

rh, sh, mh, r, t, y = sp.symbols('rho_h sigma_h mu_h r t y', positive=True)
rho, sig, mu, s0, s1, a, b = sp.symbols('rho sigma mu s0 s1 a b', positive=True)

print("== A. Receiver: uniform share formula for all three targets ==")
for name, X in [("X=t", t), ("X=(1+r)t", (1+r)*t), ("X=1+rt", 1+r*t)]:
    uR = rh*(1+r*t) - sh*(1 - t + y) - (mh/2)*(y/X - sp.Rational(1,2))**2
    ystar = sp.solve(sp.diff(uR, y), y)[0]
    ok(f"y*/X = 1/2 - (sh/mh)*X  [{name}]", sp.simplify(ystar/X - (sp.Rational(1,2) - (sh/mh)*X)) == 0)
    ok(f"receiver SOC < 0        [{name}]", sp.simplify(sp.diff(uR, y, 2)*X**2) == -mh)

print("== B. Spec (i),(ii) schedules affine decreasing; spec (iii) pathology ==")
s_i  = sp.simplify((t*(sp.Rational(1,2)-(sh/mh)*t))/((1+r)*t))            # share of output, X=t
s_ii = sp.Rational(1,2) - (sh/mh)*(1+r)*t                                  # share of output, X=(1+r)t
ok("spec (i):  s(t) = [1/2-(sh/mh)t]/(1+r), linear decreasing", sp.simplify(s_i - (sp.Rational(1,2)-(sh/mh)*t)/(1+r)) == 0 and sp.diff(s_i, t) == -sh/(mh*(1+r)))
ok("spec (ii): ds/dt = -(sh/mh)(1+r) < 0", sp.diff(s_ii, t) == -(sh/mh)*(1+r))
X3 = 1+r*t; y3 = X3*(sp.Rational(1,2)-(sh/mh)*X3)
ok("spec (iii): at t->0 target return 1/2-sh/mh > pot 0 (binds when sh/mh<1/2)", sp.simplify(sp.limit(y3 - (1+r)*t, t, 0) - (sp.Rational(1,2)-sh/mh)) == 0)

print("== C. Sender under believed s(t)=s0-s1*t ==")
u_s = rho*(1+r*t) - sig*(1+r)*t*(1-(s0-s1*t)) - (mu/2)*(t-sp.Rational(1,2))**2
den = mu + 2*sig*(1+r)*s1
num = mu/2 + rho*r - sig*(1+r)*(1-s0)
tstar = sp.solve(sp.diff(u_s, t), t)[0]
ok("SOC = -(mu + 2 sigma (1+r) s1) < 0 (global concavity)", sp.simplify(sp.diff(u_s, t, 2) + den) == 0)
ok("t* = [mu/2 + rho r - sigma(1+r)(1-s0)] / [mu + 2 sigma(1+r) s1]", sp.simplify(tstar - num/den) == 0)
ok("nests eq tg_send at s1=0", sp.simplify(tstar.subs(s1,0) - (sp.Rational(1,2) + (rho*r - sig*(1+r)*(1-s0))/mu)) == 0)
ok("dt*/drho = r/den > 0", sp.simplify(sp.diff(tstar, rho) - r/den) == 0)
dsig = sp.simplify(sp.diff(tstar, sig))
ok("dt*/dsigma < 0 unconditionally (= -(1+r)[mu(1-s0)+mu s1+2 s1 rho r]/den^2)",
   sp.simplify(dsig + (1+r)*(mu*(1-s0) + mu*s1 + 2*s1*rho*r)/den**2) == 0)
ok("dt*/dmu = -(t*-1/2)/den  (norm still pulls to equal split)", sp.simplify(sp.diff(tstar, mu) + (tstar - sp.Rational(1,2))/den) == 0)
ok("dt*/ds0 = sigma(1+r)/den  (damped level sensitivity)", sp.simplify(sp.diff(tstar, s0) - sig*(1+r)/den) == 0)
ok("dt*/ds1 = -2 sigma(1+r) t*/den", sp.simplify(sp.diff(tstar, s1) + 2*sig*(1+r)*tstar/den) == 0)

print("== D. UG rho channel (NG Commento 1) ==")
u_ug = (a+b*t)*(rho - sig*t) - (mu/2)*(t-sp.Rational(1,2))**2
t_ug = sp.solve(sp.diff(u_ug, t), t)[0]
ok("dt_UG/drho = b/(2b sigma + mu), zero at b=0", sp.simplify(sp.diff(t_ug, rho) - b/(2*b*sig+mu)) == 0)

print("== E. Calibration inversions (symbolic) ==")
x, R = sp.symbols('x R', positive=True)   # x = sigma/mu, R = rho/mu
tDG, tTG, tUG, sbar, pref = sp.symbols('t_DG t_TG t_UG s_bar p_ref', positive=True)
ok("sigma/mu = 1/2 - t_DG", sp.solve(sp.Eq(tDG, sp.Rational(1,2) - x), x)[0] == sp.Rational(1,2) - tDG)
R_sol = sp.solve(sp.Eq(tTG, sp.Rational(1,2) + (R*r - x*(1+r)*(1-sbar))), R)[0]  # mu-normalized
ok("rho/mu = [t_TG - 1/2 + (sigma/mu)(1+r)(1-sbar)]/r", sp.simplify(R_sol - (tTG - sp.Rational(1,2) + x*(1+r)*(1-sbar))/r) == 0)

print("== F. First-pass calibration, audited control moments, r=2 ==")
import numpy as np
rn = 2.0
cats = {"Moral":        dict(tDG=0.462, tUG=0.500, tTG=0.441, sbar=0.315),
        "Self-interest":dict(tDG=0.103, tUG=0.417, tTG=0.295, sbar=0.220)}
p13 = 0.483   # UG hypothetical acceptance at t=1/3, flat across categories (audited v1)
for c, m in cats.items():
    xv = 0.5 - m['tDG']
    Rv = (m['tTG'] - 0.5 + xv*(1+rn)*(1-m['sbar']))/rn
    # implied (a,b): UG FOC (mu=1):  (tUG-1/2)(2 b x + 1) = b R - (a+b) x,  a = p13 - b/3
    bv = sp.symbols('bv')
    eq = sp.Eq((m['tUG']-0.5)*(2*bv*xv+1), bv*Rv - ((p13 - bv/3) + bv)*xv)
    bsol = [sp.N(s) for s in sp.solve(eq, bv)]
    bnum = float(bsol[0]); anum = p13 - bnum/3
    interior_tg = abs(Rv*rn - xv*(1+rn)*(1-m['sbar']))  # |N|/mu, needs < 1/2
    print(f"  {c:14s} sigma/mu={xv:6.3f}  rho/mu={Rv:6.3f}  (rho vs sigma: {Rv:.3f} vs {xv:.3f})")
    print(f"  {'':14s} implied UG schedule: a={anum:6.3f}, b={bnum:6.3f}, a+b={anum+bnum:6.3f}"
          f"   TG |N|/mu={interior_tg:.3f} (<0.5 interior: {interior_tg<0.5})")
    print(f"  {'':14s} predicted sensitivities: UG x/(2bx+1)={xv/(2*bnum*xv+1):5.3f}  TG 3x={3*xv:5.3f}")
# MBC: TG locus only (DG cell ~0.8% of responses)
tTG_m, sb_m = 0.635, 0.357
print("  MBC locus: rho/mu = [0.135 + (1+r)(1-sbar) x]/r =", f"[0.135 + {(1+rn)*(1-sb_m):.3f} x]/2")
xs = np.linspace(0.0, 0.5, 6)
Rs = (tTG_m - 0.5 + xs*(1+rn)*(1-sb_m))/rn
print("    x grid:", np.round(xs,2), " rho/mu:", np.round(Rs,3), " rho>sigma everywhere:", bool((Rs > xs).all()))

print("== G. Recency structure: delta->0 => description dominates ==")
delta = 0.05; ages = [1, 3, 10, 30]
Fc = sum(delta**tau for tau in ages)
print(f"  F_c (old experiences, delta={delta}) = {Fc:.4f}  vs description weight delta^0 = 1  -> fidelity dominates: {Fc < 1}")
print(f"  story adds an age-0 trace: F_c -> {Fc+1:.4f}, now rivals the description")
