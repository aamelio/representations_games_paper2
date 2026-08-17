#!/usr/bin/env python3
"""Plug-in walkthrough of the headline calibration (post-meeting task 2 follow-up).

Backs the calibration-walkthrough appendix of post_meeting_tasks.tex: for every entry
of the published Table 4 (calibration_p1.tex), prints the general formula with the
control-sample inputs substituted, and asserts the full-precision result matches the
published value. Inputs displayed to four decimals; arithmetic at full precision.

Reuses the headline code (08_calibration.py) for data, moments, and calibration, so a
divergence between this walkthrough and the published table is impossible by
construction unless the formulas transcribed here drift from 08's.

Output: output/tables/calibration_walkthrough.txt
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("calib", HERE / "08_calibration.py")
calib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calib)

R, REF = calib.R, calib.REF

# Published Table 4 (calibration_p1.tex), the values every step must reproduce.
PUB = {
    "Moral": dict(x=0.038, y=0.022, s=0.315, imp=0.565, meas=0.795),
    "Self-interest": dict(x=0.397, y=0.426, a=0.324, b=0.481, s=0.220, imp=0.525, meas=0.701),
    "Mutual Benefit / Cooperation": dict(x=0.000, y=0.048, s=0.357, imp=0.598, meas=0.711),
}
TOL = 5e-4

LOG: list[str] = []


def log(*a) -> None:
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def check(label: str, got: float, pub: float) -> None:
    assert abs(got - pub) < TOL, f"{label}: computed {got:.4f} != published {pub:.3f}"
    log(f"    -> {got:.4f}, published {pub:.3f}  OK")


def main() -> None:
    df = calib.load_control()
    m_all = calib.moments(df)
    res = calib.calibrate(m_all)
    mo, si, mbc = (m_all[c] for c in calib.CATS)
    r_mo, r_si, r_mbc = (res[c] for c in calib.CATS)

    log("Plug-in walkthrough of the headline calibration (r = 2, reference offer 1/3)")
    log("")
    log("Inputs: control moments by category (means; p_hp/s_hp = reference-action beliefs)")
    cols = ["t_dgkw", "t_ug", "t_tg", "p_hp", "t_chosen", "p_chosen", "s_hp",
            "n_dgkw", "n_ug", "n_tg"]
    import pandas as pd
    log(pd.DataFrame(m_all).T[cols].round(4).to_string())
    log("")

    log("B.1  sigma/mu = 1/2 - t_DG   (interior DG first-order condition)")
    for cat, m in [("Moral", mo), ("Self-interest", si)]:
        x = 0.5 - m["t_dgkw"]
        log(f"  {cat}: 1/2 - {m['t_dgkw']:.4f} = {x:.4f}")
        check(f"{cat} x", x, PUB[cat]["x"])
    log("")

    log("B.2  anchor te(s) = 1 / (1 + (1+r)(1-2s))   at s = s_hp")
    tes = {}
    for cat, m in [("Moral", mo), ("Self-interest", si), ("Mutual Benefit / Cooperation", mbc)]:
        s = m["s_hp"]
        tes[cat] = calib.te(s)
        log(f"  {cat}: s = {s:.4f} -> te = 1/(1 + 3*{1 - 2 * s:.4f}) = {tes[cat]:.4f}")
    log("")

    log("B.3  rho/mu = [ t_TG - te(s) + (sigma/mu)(1+r)(1-s) ] / r")
    for cat, m in [("Moral", mo), ("Self-interest", si)]:
        x = 0.5 - m["t_dgkw"]
        y = calib.tg_locus(m, x)
        log(f"  {cat}: [ {m['t_tg']:.4f} - {tes[cat]:.4f} + "
            f"{x:.4f}*3*{1 - m['s_hp']:.4f} ] / 2 = {y:.4f}")
        check(f"{cat} y", y, PUB[cat]["y"])
    log("")

    log("B.4  (a,b) from the UG FOC + reference belief point:")
    log("     b = -[ (t_UG - 1/2) + p_hp * x ] / [ 2(t_UG - 1/2)x - y + (2/3)x ],")
    log("     a = p_hp - b/3")
    for cat, m, p in [("Self-interest", si, r_si), ("Moral", mo, r_mo)]:
        tp = m["t_ug"] - 0.5
        num = -tp - m["p_hp"] * p["x"]
        den = 2 * tp * p["x"] - p["y"] + (2.0 / 3.0) * p["x"]
        b = num / den
        a = m["p_hp"] - b / 3.0
        log(f"  {cat}: b = -[ {tp:.4f} + {m['p_hp']:.4f}*{p['x']:.4f} ] / "
            f"[ 2*{tp:.4f}*{p['x']:.4f} - {p['y']:.4f} + (2/3)*{p['x']:.4f} ]"
            f" = {num:.4f}/{den:.4f} = {b:.3f}")
        log(f"         a = {m['p_hp']:.4f} - {b:.3f}/3 = {a:.3f}")
        if cat == "Self-interest":
            check("SI a", a, PUB[cat]["a"])
            check("SI b", b, PUB[cat]["b"])
        else:
            assert b <= 0, "Moral schedule unexpectedly valid"
            log("    -> b < 0: no valid schedule; unidentified, as norm domination implies")
    log("")

    log("B.5  MBC: impose (a,b) = SI schedule; solve the UG and TG FOCs jointly in x:")
    log("     implied UG offer t(x) = 1/2 + [ b*y(x) - (a+b)x ] / (2bx + 1),")
    log("     y(x) = [ t_TG - te(s) + x(1+r)(1-s) ] / r;  find t(x) = t_UG on x in [0, 1/2)")
    a_c, b_c = r_si["a_foc"], r_si["b_foc"]
    y0 = calib.tg_locus(mbc, 0.0)
    t0 = calib.ug_foc_offer(0.0, y0, a_c, b_c)
    t45 = calib.ug_foc_offer(0.45, calib.tg_locus(mbc, 0.45), a_c, b_c)
    log(f"  at x = 0:    y(0) = [ {mbc['t_tg']:.4f} - {tes['Mutual Benefit / Cooperation']:.4f} ]"
        f" / 2 = {y0:.4f};  t(0) = 1/2 + {b_c:.3f}*{y0:.4f} = {t0:.4f}")
    log(f"  observed t_UG = {mbc['t_ug']:.4f} > t(0) = {t0:.4f}, and t(x) is decreasing in x"
        f" (t(0.45) = {t45:.4f}),")
    log("  so no interior root exists: the constrained solution is the boundary x = 0")
    check("MBC x", r_mbc["x"], PUB["Mutual Benefit / Cooperation"]["x"])
    check("MBC y", r_mbc["y"], PUB["Mutual Benefit / Cooperation"]["y"])
    log("")

    log("B.6  s column: mean believed returned share at the reference send (s_hp)")
    for cat in calib.CATS:
        check(f"{cat} s", m_all[cat]["s_hp"], PUB[cat]["s"])
    log("")

    log("B.7  overidentifying test: implied acceptance at the chosen offer = a + b*t_chosen")
    log("     (Moral uses the SI schedule, its own being unidentified); measured = p_chosen")
    for cat in calib.CATS:
        m, p = m_all[cat], res[cat]
        log(f"  {cat}: {a_c:.3f} + {b_c:.3f}*{m['t_chosen']:.4f} = {p['p_chosen_pred']:.4f};"
            f"  measured {m['p_chosen']:.4f}")
        check(f"{cat} implied", p["p_chosen_pred"], PUB[cat]["imp"])
        check(f"{cat} measured", m["p_chosen"], PUB[cat]["meas"])
    log("")

    # Section 1's belief-sensitivity table (17_tg_anchor_dryrun.py block (v))
    PUB_SENS = {
        "Moral": dict(old=0.113, new=1.461, est=0.306),
        "Self-interest": dict(old=1.192, new=2.029, est=0.451),
        "Mutual Benefit / Cooperation": dict(old=0.000, new=1.742, est=0.264),
    }
    slopes = calib.empirical_slopes(df)
    log("B.8  Section 1's belief-sensitivity table:")
    log("     pred_old = (1+r)x;  pred_new = dte(s) + (1+r)x,")
    log("     dte(s) = 2(1+r) / [1+(1+r)(1-2s)]^2;  estimated = OLS slope of the TG")
    log("     send on the hypothetical believed share, by category (model-free)")
    for cat in calib.CATS:
        m, p = m_all[cat], res[cat]
        old = (1 + R) * p["x"]
        d = calib.dte(m["s_hp"])
        new = d + old
        est, se, n = slopes[("tg", cat)]
        log(f"  {cat}: old = 3*{p['x']:.4f} = {old:.4f};"
            f"  dte({m['s_hp']:.4f}) = 6/{1 + 3 * (1 - 2 * m['s_hp']):.4f}^2 = {d:.4f};"
            f"  new = {d:.4f}+{old:.4f} = {new:.4f}")
        r_old = f"{est / old:.2f}" if old > 0 else "inf (old = 0)"
        log(f"    estimated (OLS) = {est:.4f} (se {se:.3f}, N {n});"
            f"  est/old = {r_old};  est/new = {est / new:.2f}")
        check(f"{cat} pred old", old, PUB_SENS[cat]["old"])
        check(f"{cat} pred new", new, PUB_SENS[cat]["new"])
        check(f"{cat} estimated", est, PUB_SENS[cat]["est"])
    log("")
    log("All entries of the published Table 4 and of Section 1's sensitivity table")
    log("reproduced from the stated formulas.")

    out = calib.TABLES / "calibration_walkthrough.txt"
    out.write_text("\n".join(LOG))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
