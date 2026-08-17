import sympy as sp
ok = lambda name, cond: print(("  PASS  " if cond else "  FAIL  ") + name)

rho, sig, mu, a, b, r, s, sa, t = sp.symbols('rho sigma mu a b r s s_a t', positive=True)

# Optima (interior)
t_dg = sp.Rational(1,2) - sig/mu
t_ug = sp.Rational(1,2) + (b*rho-(a+b)*sig)/(2*b*sig+mu)
t_tg = sp.Rational(1,2) + (rho*r - sig*(1+r)*(1-s))/mu

print("== WELFARE: V_DG = 1, V_UG = p(t*), V_TG = 1 + r t* ==")
V_ug = a + b*t_ug
V_tg = 1 + r*t_tg
ok("dV_UG/drho = b^2/(2b sig + mu)",      sp.simplify(sp.diff(V_ug, rho) - b**2/(2*b*sig+mu)) == 0)
ok("dV_TG/drho = r^2/mu",                 sp.simplify(sp.diff(V_tg, rho) - r**2/mu) == 0)
ok("dV_UG/dsig = -b[(a+b)mu+2b^2 rho]/(2b sig+mu)^2 < 0",
   sp.simplify(sp.diff(V_ug, sig) + b*((a+b)*mu+2*b**2*rho)/(2*b*sig+mu)**2) == 0)
ok("dV_TG/dsig = -r(1+r)(1-s)/mu < 0",    sp.simplify(sp.diff(V_tg, sig) + r*(1+r)*(1-s)/mu) == 0)
ok("dV_UG/dmu = -b(t*-1/2)/(2b sig+mu): sign of (1/2 - t*)",
   sp.simplify(sp.diff(V_ug, mu) + b*(t_ug-sp.Rational(1,2))/(2*b*sig+mu)) == 0)
ok("dV_TG/dmu = -r(t*-1/2)/mu: sign of (1/2 - t*)",
   sp.simplify(sp.diff(V_tg, mu) + r*(t_tg-sp.Rational(1,2))/mu) == 0)
ok("dV_UG/da = (b sig + mu)/(2b sig + mu) in (1/2, 1)",
   sp.simplify(sp.diff(V_ug, a) - (b*sig+mu)/(2*b*sig+mu)) == 0)
ok("dV_TG/ds = r sig (1+r)/mu > 0",       sp.simplify(sp.diff(V_tg, s) - r*sig*(1+r)/mu) == 0)

print("== Counterpart's expected payoff: sigma hurts unambiguously ==")
piR_ug = (a+b*t_ug)*t_ug
ok("dpiR_UG/dsig = (b t + p) dt/dsig < 0", sp.simplify(sp.diff(piR_ug, sig) - (b*t_ug+(a+b*t_ug))*sp.diff(t_ug,sig)) == 0
   and bool(sp.diff(t_ug, sig).subs({a:sp.Rational(1,4),b:sp.Rational(1,2),rho:1,sig:1,mu:1}) < 0))
piR_tg = (1-sa)*(1+r)*t_tg
ok("dpiR_TG/dsig = (1-s_a)(1+r) dt/dsig < 0", sp.simplify(sp.diff(piR_tg, sig) - (1-sa)*(1+r)*sp.diff(t_tg, sig)) == 0)
# Sender's own expected payoff in UG: ambiguous in sigma (both signs attainable)
piS_ug = (a+b*t_ug)*(1-t_ug)
d1 = sp.diff(piS_ug, sig).subs({a:sp.Rational(1,4), b:sp.Rational(2,3), rho:sp.Rational(2,5), mu:1, sig:sp.Rational(2,5)})
d2 = sp.diff(piS_ug, sig).subs({a:sp.Rational(9,10), b:sp.Rational(1,20), rho:1, mu:sp.Rational(1,10), sig:sp.Rational(1,100)})
print(f"    dpiS_UG/dsig examples: {sp.N(d1,3)} and {sp.N(d2,3)}  -> ambiguous: {(d1>0) != (d2>0)}")

print("== INEQUALITY (conditional on trade): D = pi_S - pi_R ==")
D_dg = 1 - 2*t_dg; D_ug = 1 - 2*t_ug
D_tg = (1 - t + sa*(1+r)*t) - (1-sa)*(1+r)*t
k = sp.symbols('k'); kval = 1 + (1+r)*(1-2*sa)
ok("D_DG = 2 sig/mu",                     sp.simplify(D_dg - 2*sig/mu) == 0)
ok("D_UG = 2[(a+b)sig - b rho]/(2b sig+mu)", sp.simplify(D_ug - 2*((a+b)*sig-b*rho)/(2*b*sig+mu)) == 0)
ok("D_TG(t) = 1 - t[1+(1+r)(1-2 s_a)]",   sp.simplify(D_tg - (1 - t*kval)) == 0)
ok("dD_UG/dsig > 0, dD_UG/drho < 0",
   sp.simplify(sp.diff(D_ug, sig) - 2*((a+b)*mu+2*b**2*rho)/(2*b*sig+mu)**2) == 0
   and sp.simplify(sp.diff(D_ug, rho) + 2*b/(2*b*sig+mu)) == 0)
ok("dD_DG/dmu = -D/mu, dD_UG/dmu = -D/(2b sig+mu) (compression to zero)",
   sp.simplify(sp.diff(D_dg, mu) + D_dg/mu) == 0 and sp.simplify(sp.diff(D_ug, mu) + D_ug/(2*b*sig+mu)) == 0)
D_tg_star = (1 - t_tg*kval)
ok("dD_TG/dmu = -(D - D(1/2))/mu (compression to equal-split value)",
   sp.simplify(sp.diff(D_tg_star, mu) + (D_tg_star - (1-kval/2))/mu) == 0)
ok("D_TG(1/2) = 0 iff s_a = r/(2(1+r))",  sp.simplify(sp.solve(sp.Eq(1-kval/2, 0), sa)[0] - r/(2*(1+r))) == 0)
print("    at r=2: equal-split send equalizes payoffs iff s_a =", sp.nsimplify(r/(2*(1+r))).subs(r,2))

print("== RECEIVER: composition steepens the acceptance schedule ==")
# two categories: Moral bad (p_MB flat low), Self-interest (p_SI = a + b t); retrieval q_SI(t) rises in t
import numpy as np
tt = np.linspace(0.05, 0.5, 10)
p_MB = 0.05 + 0*tt
p_SI = 0.20 + 0.70*tt
q_SI = 1/(1+np.exp(-8*(tt-0.25)))          # low offers retrieve Moral bad
p_agg = q_SI*p_SI + (1-q_SI)*p_MB
slope_agg = np.gradient(p_agg, tt)
print(f"    max category slope = 0.70; aggregate slope range = [{slope_agg.min():.2f}, {slope_agg.max():.2f}]"
      f" -> steeper than any category: {slope_agg.max() > 0.70}")
# symbolic decomposition p' = sum q p' + sum q' p
q = sp.Function('q')(t); pA = sp.Function('p_A')(t); pB = sp.Function('p_B')(t)
p_mix = q*pA + (1-q)*pB
ok("p'(t) = [q pA' + (1-q) pB'] + q'(pA - pB)  (within + composition)",
   sp.simplify(sp.diff(p_mix, t) - (q*sp.diff(pA,t)+(1-q)*sp.diff(pB,t) + sp.diff(q,t)*(pA-pB))) == 0)
