import sympy as sp
ok = lambda n, c: print(("  PASS  " if c else "  FAIL  ") + n)

# --- rem:foundation: accept-reject margin and uniform -> linear schedule
rh, sh, mh, t, psi, Psi = sp.symbols('rho_h sigma_h mu_h t psi Psi', positive=True)
ineq = sp.Symbol('I')  # idealistic inequity cost of the offer, felt in BOTH branches
u_accept = rh*1 - sh*(1-t) - ineq
u_reject = rh*0 - sh*0 + mh*psi - ineq
margin = sp.simplify(u_accept - u_reject)
ok("accept-reject margin = rho_h - sigma_h(1-t) - mu_h psi (inequity cancels)",
   sp.simplify(margin - (rh - sh*(1-t) - mh*psi)) == 0)
# psi ~ U[0, Psi]: P(accept) = P(psi <= (rho_h - sigma_h + sigma_h t)/mu_h) = a + b t on the linear region
a_f = (rh-sh)/(mh*Psi); b_f = sh/(mh*Psi)
ok("uniform psi gives a=(rho_h-sigma_h)/(mu_h Psi), b=sigma_h/(mu_h Psi)",
   sp.simplify((rh - sh + sh*t)/(mh*Psi) - (a_f + b_f*t)) == 0)

# --- welfare (ii): exact condition for the sender's own expected payoff to fall in sigma
rho, sig, mu, a, b = sp.symbols('rho sigma mu a b', positive=True)
t_ug = sp.Rational(1,2) + (b*rho-(a+b)*sig)/(2*b*sig+mu)
piS = (a+b*t_ug)*(1-t_ug)
lhs = sp.diff(piS, sig)
factored = (b*(1-t_ug) - (a+b*t_ug)) * sp.diff(t_ug, sig)
ok("dpiS/dsigma = [b(1-t) - p(t)] dt/dsigma (chain rule form)", sp.simplify(lhs - factored) == 0)
ok("b(1-t) - p(t) = b(1-2t) - a  (so dpiS/dsigma < 0 iff b(1-2t*) > a, since dt/dsigma<0)",
   sp.simplify((b*(1-t) - (a+b*t)) - (b*(1-2*t) - a)) == 0)
w = {a:sp.Rational(1,20), b:sp.Rational(3,5), rho:sp.Rational(1,10), sig:1, mu:1}
ok("witness satisfies b(1-2t*)>a", bool((b*(1-2*t_ug)-a).subs(w) > 0))

# --- welfare (iii) scope: false for DG (V_DG constant, yet 1/2 - t_DG > 0)
t_dg = sp.Rational(1,2) - sig/mu
ok("DG: dV/dmu = 0 but 1/2 - t_DG = sigma/mu > 0 -> claim needs 'strategic games' scope",
   sp.diff(sp.Integer(1), mu) == 0 and sp.simplify(sp.Rational(1,2)-t_dg - sig/mu) == 0)

# --- prop:ug interiority (2026-07-13): censored solution in (0,1) iff |b rho - (a+b) sigma| < b sigma + mu/2
tf = sp.Symbol('t_f')
v_ug = (a + b*tf)*(rho - sig*tf) - (mu/2)*(tf - sp.Rational(1,2))**2
t_star = sp.solve(sp.diff(v_ug, tf), tf)[0]
ok("UG FOC solution equals eq:ug_offer", sp.simplify(t_star - t_ug) == 0)
num_ug = b*rho - (a+b)*sig; half_curv = b*sig + mu/2
ok("threshold = half the curvature |v''|: b sigma + mu/2 = (2b sigma + mu)/2",
   sp.simplify(-sp.diff(v_ug, tf, 2)/2 - half_curv) == 0)
ok("t* > 0 iff num > -(b sigma + mu/2)",
   sp.simplify(t_star - (half_curv + num_ug)/(2*b*sig+mu)) == 0)
ok("t* < 1 iff num < +(b sigma + mu/2)",
   sp.simplify((1 - t_star) - (half_curv - num_ug)/(2*b*sig+mu)) == 0)
# both corners reachable under the maintained assumptions (witnesses with a,b>0, a+b<=1) ...
up_w = {a:sp.Rational(1,20), b:sp.Rational(9,10), rho:2, sig:sp.Rational(1,10), mu:sp.Rational(1,10)}
lo_w = {a:sp.Rational(9,10), b:sp.Rational(1,20), rho:sp.Rational(1,10), sig:2, mu:sp.Rational(1,2)}
ok("upper-corner witness (a+b<=1): t* > 1", bool(t_star.subs(up_w) > 1))
ok("lower-corner witness (a+b<=1): t* < 0", bool(t_star.subs(lo_w) < 0))
# ... but discipline shields the zero corner: the UNCONSTRAINED gap is positive under a+b<=1,
# so t*_UG <= 0 requires t*_DG < 0 (the UG censors at 0 only where the DG already does)
ok("unconstrained t*_UG - t*_DG = [mu b rho + 2b sigma^2 + mu sigma(1-a-b)]/[mu(2b sigma+mu)]",
   sp.simplify(t_ug - t_dg - (mu*b*rho + 2*b*sig**2 + mu*sig*(1-a-b))/(mu*(2*b*sig+mu))) == 0)

# --- receiver composition positivity with >2 categories (mass leaves lowest-p category)
qMB, q2, q3, pMB, p2, p3, dq2, dq3 = sp.symbols("q_MB q2 q3 p_MB p2 p3 dq2 dq3", positive=True)
comp = (-(dq2+dq3))*pMB + dq2*p2 + dq3*p3   # sum q_c' p_c with q_MB' = -(dq2+dq3)
ok("composition = dq2(p2-p_MB) + dq3(p3-p_MB) > 0 when others' p exceed Moral bad's",
   sp.simplify(comp - (dq2*(p2-pMB) + dq3*(p3-pMB))) == 0)

# --- prop:tg_endog (2026-07-13): trust send under the declining schedule s(t) = s0 - s1*t
s0, s1, r = sp.symbols('s0 s1 r', positive=True)
w_tg = rho*(1 + r*tf) - sig*(1+r)*(1 - s0 + s1*tf)*tf - (mu/2)*(tf - sp.Rational(1,2))**2
D_tg = mu + 2*sig*(1+r)*s1
N_tg = mu/2 + rho*r - sig*(1+r)*(1-s0)
t_tge = sp.solve(sp.diff(w_tg, tf), tf)[0]
ok("TG-endog FOC solution equals N/D (eq:tg_send_endog)", sp.simplify(t_tge - N_tg/D_tg) == 0)
ok("concavity: w'' = -(mu + 2 sigma (1+r) s1)", sp.simplify(sp.diff(w_tg, tf, 2) + D_tg) == 0)
ok("nests eq:tg_send at s1=0",
   sp.simplify(t_tge.subs(s1, 0) - (sp.Rational(1,2) + (rho*r - sig*(1+r)*(1-s0))/mu)) == 0)
ok("t - 1/2 = [rho r - sigma(1+r)(1-s0+s1)]/D (so interior iff |num| < D/2 = mu/2 + sigma(1+r)s1)",
   sp.simplify((t_tge - sp.Rational(1,2)) - (rho*r - sig*(1+r)*(1-s0+s1))/D_tg) == 0)
ok("dt/ds0 = sigma(1+r)/D (level sensitivity, damped by the believed slope)",
   sp.simplify(sp.diff(t_tge, s0) - sig*(1+r)/D_tg) == 0)
ok("dt/ds1 = -2 sigma (1+r) t/D", sp.simplify(sp.diff(t_tge, s1) + 2*sig*(1+r)*t_tge/D_tg) == 0)
ok("dt/dmu = -(t - 1/2)/D (toward the anchor)",
   sp.simplify(sp.diff(t_tge, mu) + (t_tge - sp.Rational(1,2))/D_tg) == 0)
ok("dt/dsigma = -(1+r)[(1-s0)D + 2 s1 N]/D^2 (< 0 at interior optima, where N > 0)",
   sp.simplify(sp.diff(t_tge, sig) + (1+r)*((1-s0)*D_tg + 2*s1*N_tg)/D_tg**2) == 0)
# rem:return foundation: receiver as dictator over target X returns y*/X = 1/2 - (sigma_h/mu_h) X
X, y = sp.symbols('X y', positive=True)
u_recv = rh*(1 + r*t) - sh*(1 - t + y) - (mh/2)*(y/X - sp.Rational(1,2))**2
ystar = sp.solve(sp.diff(u_recv, y), y)[0]
ok("receiver returns y*/X = 1/2 - (sigma_h/mu_h) X for any target X",
   sp.simplify(ystar/X - (sp.Rational(1,2) - (sh/mh)*X)) == 0)
# rem:return, two tractable targets (2026-07-14): believed returned share OF THE OUTPUT, s(t) = y*/((1+r)t)
s_sent = ystar.subs(X, t)/((1+r)*t)
s_out = ystar.subs(X, (1+r)*t)/((1+r)*t)
ok("amount-sent target (X=t): s(t) = [1/2 - (sigma_h/mu_h) t]/(1+r)",
   sp.simplify(s_sent - (sp.Rational(1,2) - (sh/mh)*t)/(1+r)) == 0)
ok("output target (X=(1+r)t): s(t) = 1/2 - (sigma_h/mu_h)(1+r) t",
   sp.simplify(s_out - (sp.Rational(1,2) - (sh/mh)*(1+r)*t)) == 0)
ok("both schedules affine (s'' = 0) and decreasing in the send",
   sp.simplify(sp.diff(s_sent, t, 2)) == 0 and sp.simplify(sp.diff(s_out, t, 2)) == 0
   and bool(sp.simplify(-sp.diff(s_sent, t)).is_positive) and bool(sp.simplify(-sp.diff(s_out, t)).is_positive))
ok("prop:tg_endog covers both: (s0,s1) = (1/(2(1+r)), (sh/mh)/(1+r)) and (1/2, (sh/mh)(1+r)), with s0 < 1",
   sp.simplify(s_sent - (1/(2*(1+r)) - (sh/mh)/(1+r)*t)) == 0
   and bool(sp.simplify(1 - 1/(2*(1+r))).is_positive) and bool(sp.Rational(1,2) < 1))
# tractability: for X=t and X=(1+r)t the optimum y* <= X/2 never exceeds the pot (1+r)t ...
ok("feasibility: X/2 < (1+r)t for X=t and X=(1+r)t (upper constraint never binds)",
   bool(sp.simplify((1+r)*t - t/2).is_positive) and bool(sp.simplify((1+r)*t - (1+r)*t/2).is_positive))
# ... while the total-wealth norm target (1+rt)/2 exceeds the pot iff t < 1/(2+r): the kink
ok("total-wealth target: (1+rt)/2 - (1+r)t = (1 - (2+r)t)/2, positive iff t < 1/(2+r)",
   sp.simplify((1 + r*t)/2 - (1+r)*t - (1 - (2+r)*t)/2) == 0)

# --- welfare, believed vs realized (2026-07-13): (i)-(iii) robust to an actual schedule, (iv) reverses
ah, bh, s = sp.symbols('a_hat b_hat s', positive=True)
V_bel = a + b*t_ug                       # V_UG = p(t*) under the sender's own schedule
ok("believed dV_UG/da = (b sigma + mu)/(2b sigma + mu) > 0",
   sp.simplify(sp.diff(V_bel, a) - (b*sig + mu)/(2*b*sig + mu)) == 0)
V_act = ah + bh*t_ug                     # realized V under an independent actual schedule p_hat
ok("realized dV_UG/da = -b_hat sigma/(2b sigma + mu) < 0 (part (iv) REVERSES)",
   sp.simplify(sp.diff(V_act, a) + bh*sig/(2*b*sig + mu)) == 0)
for th, nm in [(rho, 'rho'), (sig, 'sigma'), (mu, 'mu')]:
    ok(f"realized dV_UG/d{nm} = b_hat * dt/d{nm} (signs of (i)-(iii) carry)",
       sp.simplify(sp.diff(V_act, th) - bh*sp.diff(t_ug, th)) == 0)
t_tgc = sp.Rational(1,2) + (rho*r - sig*(1+r)*(1-s))/mu
ok("TG realized dV/ds = r sigma(1+r)/mu > 0 (no flip; total wealth is division-free)",
   sp.simplify(sp.diff(1 + r*t_tgc, s) - r*sig*(1+r)/mu) == 0)
