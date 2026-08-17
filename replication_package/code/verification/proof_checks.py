import sympy as sp

t, rho, sig, mu, a, b, r, s = sp.symbols('t rho sigma mu a b r s', positive=True)

# --- DG ---
u = rho - sig*t - mu/2*(t-sp.Rational(1,2))**2
tdg = sp.solve(sp.diff(u,t), t)[0]
assert sp.simplify(tdg - (sp.Rational(1,2) - sig/mu)) == 0
assert sp.diff(u,t,2) == -mu

# --- UG ---
v = (a+b*t)*(rho-sig*t) - mu/2*(t-sp.Rational(1,2))**2
assert sp.simplify(sp.diff(v,t,2) + (2*b*sig+mu)) == 0
tug = sp.solve(sp.diff(v,t), t)[0]
tug_paper = sp.Rational(1,2) + (b*rho-(a+b)*sig)/(2*b*sig+mu)
assert sp.simplify(tug - tug_paper) == 0
# (i) difference vs interior DG
diff_ug = sp.simplify(tug_paper - (sp.Rational(1,2)-sig/mu))
target = (mu*b*rho + 2*b*sig**2 + mu*sig*(1-a-b))/(mu*(2*b*sig+mu))
assert sp.simplify(diff_ug - target) == 0
# (ii)
assert sp.simplify(sp.diff(tug_paper,rho) - b/(2*b*sig+mu)) == 0
assert sp.simplify(sp.diff(tug_paper,sig) + ((a+b)*mu+2*b**2*rho)/(2*b*sig+mu)**2) == 0
assert sp.simplify(sp.diff(tug_paper,mu) + (tug_paper-sp.Rational(1,2))/(2*b*sig+mu)) == 0
# (iii)
assert sp.simplify(sp.diff(tug_paper,a) + sig/(2*b*sig+mu)) == 0
sens = sig/(2*b*sig+mu)
assert sp.simplify(sp.diff(sens,sig) - mu/(2*b*sig+mu)**2) == 0
assert sp.simplify(sp.diff(sens,mu) + sig/(2*b*sig+mu)**2) == 0

# --- TG ---
w = rho*(1+r*t) - sig*(1+r)*(1-s)*t - mu/2*(t-sp.Rational(1,2))**2
ttg = sp.solve(sp.diff(w,t), t)[0]
ttg_paper = sp.Rational(1,2) + (rho*r - sig*(1+r)*(1-s))/mu
assert sp.simplify(ttg - ttg_paper) == 0
assert sp.simplify(sp.diff(ttg_paper,rho) - r/mu) == 0
assert sp.simplify(sp.diff(ttg_paper,s) - sig*(1+r)/mu) == 0
assert sp.simplify(sp.diff(ttg_paper,sig) + (1+r)*(1-s)/mu) == 0
assert sp.simplify(sp.diff(ttg_paper,mu) + (ttg_paper-sp.Rational(1,2))/mu) == 0
# mu->0 slope for purely material sender (rho=sigma) via pi_S
piS = 1 + r*t - (1-s)*(1+r)*t
assert sp.simplify(sp.expand(piS) - (1 + t*(s*(1+r)-1))) == 0
assert sp.simplify((rho*r - sig*(1+r)*(1-s)).subs(rho,sig) - sig*(s*(1+r)-1)) == 0
# difference vs interior DG, and knife-edge
diff_tg = sp.simplify(ttg_paper - (sp.Rational(1,2)-sig/mu))
target_tg = (r*(rho-sig*(1-s)) + sig*s)/mu
assert sp.simplify(diff_tg - target_tg) == 0
assert sp.simplify((rho*r - sig*(1+r)).subs(rho,sig) + sig) == 0  # pull = -sigma at s=0, rho=sigma

# --- unified formula ---
for M, lab in [(rho - sig*t, 'DG'), (sp.expand((a+b*t)*(rho-sig*t)), 'UG'), (rho*(1+r*t)-sig*(1+r)*(1-s)*t, 'TG')]:
    obj = M - mu/2*(t-sp.Rational(1,2))**2
    topt = sp.solve(sp.diff(obj,t), t)[0]
    M1 = sp.diff(M,t).subs(t,sp.Rational(1,2)); M2 = sp.diff(M,t,2)
    assert sp.simplify(topt - (sp.Rational(1,2) + M1/(mu-M2))) == 0, lab

print("ALL PROOF IDENTITIES VERIFIED")
