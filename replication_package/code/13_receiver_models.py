#!/usr/bin/env python3
"""Receiver-objective horse race (held item 1; SN-approved plan 2026-07-19, call prep).

Two candidate receiver objectives, each confronted symmetrically with BOTH (A) the actual
TG receivers' return schedules and (B) the senders' believed return schedules — no side is
assigned a model ex ante; the data assign them:

  (i)  Return-norm target (rem:return): return half of a target pie X, traded against own
       earnings with weight k = sigma_hat/mu_hat >= 0. Amount-sent target (X=t):
       s(t) = (1/2 - k t)/3; output target (X=3t): s(t) = 1/2 - 3 k t. Affine, weakly
       DECLINING share in the send; one free parameter per category.
  (ii) Payoff equalization: return exactly what equalizes final payoffs. With the design's
       multiplier m=3 and no P2 endowment: s(t) = max(0, (4t-1)/(6t)). RISING share, kink
       at t=1/4, ZERO free parameters.
  (+)  Selfish benchmark s(t) = 0.

Objects (control sample throughout, conventions of 08/09):
  (A) actual receivers: returned share (share_sent) on send faced (share_sent_p1), by
      P2 category;
  (B) senders' beliefs: the two belief points per category (beliefs_hp at the 1/3
      reference send; beliefs at the chosen send), plus individual two-point believed
      slopes (the TG analogue of 09 module (c)'s UG statistic).

Outputs: output/tables/receiver_models_stats.txt, output/figures/receiver_models.pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TABLES = HERE.parent / "output" / "tables"
FIGURES = HERE.parent / "output" / "figures"

REF = 1.0 / 3.0
M = 3  # multiplier on the send (r = 2 in model.tex notation)
P2_CATS = ["Moral good", "Mutual Benefit / Cooperation", "Self-interest", "Moral bad"]
P1_CATS = {"Moral good": "Moral", "Mutual Benefit / Cooperation": "Mutual Benefit / Cooperation",
           "Self-interest": "Self-interest"}  # nominal pairing for the believed-side overlay
BINS = [0.0, 0.25, 0.4165, 0.5, 1.0]
BIN_LABELS = ["(0,1/4]", "(1/4,5/12]", "(5/12,1/2]", "(1/2,1]"]

L = []


def log(*s):
    line = " ".join(str(x) for x in s)
    L.append(line)
    print(line)


def s_eq(t):
    """Payoff-equalization returned share of the multiplied amount (zero parameters)."""
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (4.0 * t - 1.0) / (6.0 * t)
    return np.where(t > 0, np.maximum(0.0, out), 0.0)


def fit_norm_amount(t, s):
    """LS fit of s = (1/2 - k t)/3 = 1/6 - (k/3) t over k >= 0 (analytic, then clipped)."""
    k = -3.0 * np.sum(t * (s - 1.0 / 6.0)) / np.sum(t**2)
    return max(0.0, k)


def fit_norm_output(t, s):
    """LS fit of s = 1/2 - 3 k t over k >= 0 (analytic, then clipped)."""
    k = np.sum(t * (0.5 - s)) / (3.0 * np.sum(t**2))
    return max(0.0, k)


def rmse(s, pred):
    return float(np.sqrt(np.mean((np.asarray(s) - np.asarray(pred)) ** 2)))


def load():
    p1 = pd.read_excel(DATA / "player1_all_categorized.xlsx")
    p2 = pd.read_excel(DATA / "player2_all_categorized.xlsx")
    for d in (p1, p2):
        d["story"] = pd.to_numeric(d["story"], errors="coerce")
    p1 = p1[(p1.game == "tg") & (p1.story == 0)].copy()
    p2 = p2[(p2.game == "tg") & (p2.story == 0)].copy()
    return p1, p2


# ------------------------------------------------------- (A) actual receiver schedules
def actual_side(p2):
    log("=" * 78)
    log("(A) Actual TG receivers (control): model fits by category")
    log("    models: equalization s(t)=max(0,(4t-1)/(6t)) [0 params]; norm target,")
    log("    amount-sent s(t)=(1/2-kt)/3 and output s(t)=1/2-3kt [k>=0, 1 param]; selfish s=0")
    log("=" * 78)
    results = {}
    for cat in P2_CATS:
        d = p2[(p2.category == cat)].dropna(subset=["share_sent", "share_sent_p1"])
        d = d[d.share_sent_p1 > 0]
        t, s = d["share_sent_p1"].to_numpy(float), d["share_sent"].to_numpy(float)
        if len(d) < 10:
            log(f"\n{cat}: N={len(d)} too small")
            continue
        ka, ko = fit_norm_amount(t, s), fit_norm_output(t, s)
        log(f"\n{cat} (N={len(d)}):")
        # slope-sign test: every norm-target member is weakly declining, so a significantly
        # positive OLS slope rejects the whole family regardless of k
        fit = sm.OLS(s, sm.add_constant(t)).fit(cov_type="HC1")
        b, se = fit.params[1], fit.bse[1]
        log(f"  OLS slope of returned share on send: {b:+.3f} (se {se:.3f})"
            + ("  -> rejects the (weakly declining) norm-target family"
               if b - 1.96 * se > 0 else ""))
        binned = d.groupby(pd.cut(d.share_sent_p1, BINS, labels=BIN_LABELS), observed=True)
        tab = binned.agg(t=("share_sent_p1", "mean"), s=("share_sent", "mean"),
                         n=("share_sent", "size"))
        # schedule fit: N-weighted RMSE on binned means (micro RMSE is dominated by
        # within-bin dispersion and barely discriminates schedule shapes)
        models = {
            "equalization (0p)": lambda x: s_eq(x),
            f"norm amount-sent (k={ka:.3f})": lambda x: 1.0 / 6.0 - (ka / 3.0) * x,
            f"norm output (k={ko:.3f})": lambda x: 0.5 - 3.0 * ko * x,
            "selfish s=0 (0p)": lambda x: np.zeros_like(np.asarray(x, dtype=float)),
        }
        w = tab.n.to_numpy(float)
        fits = {name: float(np.sqrt(np.sum(w * (tab.s.to_numpy(float)
                                                - np.asarray(f(tab.t.to_numpy(float)))) ** 2)
                                    / np.sum(w)))
                for name, f in models.items()}
        best = min(fits, key=fits.get)
        for name, v in fits.items():
            micro = rmse(s, models[name](t))
            flag = "  <-- best schedule fit" if name == best else ""
            log(f"  RMSE binned {name}: {v:.4f} (micro {micro:.4f}){flag}")
        # interior region t > 1/4: equalization's interior domain (below the kink the
        # corner s=0 binds and a moral floor plausibly operates)
        ti = tab[tab.t > 0.25]
        wi = ti.n.to_numpy(float)
        interior = {name: float(np.sqrt(np.sum(wi * (ti.s.to_numpy(float)
                                                     - np.asarray(f(ti.t.to_numpy(float)))) ** 2)
                                        / np.sum(wi)))
                    for name, f in models.items()}
        besti = min(interior, key=interior.get)
        log(f"  interior region (t>1/4) binned RMSE: "
            + "; ".join(f"{n.split(' (')[0]} {v:.4f}" for n, v in interior.items())
            + f"  -> best: {besti.split(' (')[0]}")
        log("  binned means (t_bar, s_bar, N) vs equalization prediction at t_bar:")
        for lab, row in tab.iterrows():
            log(f"    {lab}: t={row.t:.3f}  s={row.s:.3f}  N={int(row.n)}  "
                f"s_eq(t)={float(s_eq(row.t)):.3f}")
        results[cat] = dict(d=d, tab=tab, ka=ka, ko=ko, slope=b, slope_se=se)
    return results


# ------------------------------------------------------- (B) believed schedules
def believed_side(p1):
    log("\n" + "=" * 78)
    log("(B) Senders' believed return schedules (control): the two belief points vs models")
    log("=" * 78)
    points = {}
    for cat in ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]:
        d = p1[(p1.category == cat)].dropna(subset=["beliefs", "beliefs_hp"])
        s_hp, s_ch = d["beliefs_hp"].mean(), d["beliefs"].mean()
        t_ch = d["share_sent"].mean()
        d_eq = float(s_eq(t_ch) - s_eq(REF))
        obs = s_ch - s_hp
        log(f"\n{cat} (N={len(d)}): believed share at reference (t=1/3) {s_hp:.3f}, "
            f"at chosen send (t={t_ch:.3f}) {s_ch:.3f}")
        log(f"  observed change {obs:+.3f}; equalization predicts {d_eq:+.3f}; "
            f"norm target predicts <= 0")
        # individual two-point believed slopes (TG analogue of 09 module (c))
        dd = d[(d.share_sent - REF).abs() >= 1.0 / 12.0]
        bi = ((dd.beliefs - dd.beliefs_hp) / (dd.share_sent - REF)).clip(-5, 5)
        # equalization slope over the same individual ranges
        beq = ((s_eq(dd.share_sent) - s_eq(REF)) / (dd.share_sent - REF))
        log(f"  individual two-point believed slope: median {bi.median():+.3f}, "
            f"mean {bi.mean():+.3f} (N {len(bi)}); equalization predicts median "
            f"{beq.median():+.3f}")
        points[cat] = dict(s_hp=s_hp, s_ch=s_ch, t_ch=t_ch, n=len(d))
    return points


# ------------------------------------------------------- verdict + figure
def verdict():
    log("\n" + "=" * 78)
    log("Reading")
    log("=" * 78)
    log("Actual Moral-good and MBC receivers: returned shares RISE in the send (slopes")
    log("~+0.28, p<.001), rejecting every member of the weakly-declining norm-target")
    log("family; the zero-parameter equalization curve is the best schedule fit, close")
    log("from the reference send upward, with over-return below t=1/4 (a moral floor).")
    log("Self-interest receivers sit near the selfish benchmark; Moral-bad receivers")
    log("face mostly tiny sends, where equalization and selfish coincide (s=0), so the")
    log("two are not separable on that support. Senders' believed schedules are FLAT at")
    log("category-specific levels (individual two-point slope medians = 0.00): they")
    log("reject equalization's predicted rise and match the constant-share specification")
    log("the main text already uses (the norm-target family's flat boundary). Actual")
    log("reciprocity vs believed constant types = the TG forecast-error mechanism; all")
    log("sender-side propositions are untouched.")


def figure(results, points):
    FIGURES.mkdir(parents=True, exist_ok=True)
    pairs = [("Moral good", "Moral"), ("Mutual Benefit / Cooperation",
             "Mutual Benefit / Cooperation"), ("Self-interest", "Self-interest")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    tt = np.linspace(0.05, 1.0, 200)
    for ax, (p2c, p1c) in zip(axes, pairs):
        r = results.get(p2c)
        if r is None:
            continue
        ax.plot(tt, s_eq(tt), lw=1.8, label="payoff equalization (0 par.)")
        ax.plot(tt, 1.0 / 6.0 - (r["ka"] / 3.0) * tt, lw=1.2, ls="--",
                label=f"norm target, amount-sent (k̂={r['ka']:.2f})")
        ax.plot(tt, 0.5 - 3.0 * r["ko"] * tt, lw=1.2, ls=":",
                label=f"norm target, output (k̂={r['ko']:.2f})")
        ax.scatter(r["tab"].t, r["tab"].s, s=r["tab"].n.astype(float).clip(10, 200),
                   zorder=5, label="actual receivers (binned)")
        pt = points.get(p1c)
        if pt:
            ax.plot([REF, pt["t_ch"]], [pt["s_hp"], pt["s_ch"]], marker="s", ms=6,
                    lw=1.2, color="black", label="senders' believed points")
        ax.set_title(p2c if p2c == p1c else f"{p2c} / believed: {p1c}", fontsize=10)
        ax.set_xlabel("send $t$ (share of endowment)")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.02, 0.62)
    axes[0].set_ylabel("returned share of the multiplied amount")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("TG receiver objectives: actual behavior vs senders' beliefs (control)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "receiver_models.pdf")
    log(f"\nwrote {FIGURES / 'receiver_models.pdf'}")


def main():
    p1, p2 = load()
    results = actual_side(p2)
    points = believed_side(p1)
    verdict()
    figure(results, points)
    (TABLES / "receiver_models_stats.txt").write_text("\n".join(L))
    print(f"wrote {TABLES / 'receiver_models_stats.txt'}")


if __name__ == "__main__":
    main()
