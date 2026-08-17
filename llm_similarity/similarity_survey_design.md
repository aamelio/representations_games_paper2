# Similarity measurement, phase 2: LLM category round + human survey (design)

Purpose: close the circle NG asks for — the retrieval model (paper_v2 §2.2, eq. retrieval) runs on the similarity of our *contexts* to *categories of experience* and to *receiver types*; the May LLM exercise measured story↔game structural similarity (now paper_v2 §3.3, Table `tab:llm_similarity`). Two modules remain: category-level similarity, and human validation.

## Module A — LLM round 2: contexts × categories

- **Contexts (8):** Bonus story; Aid story; Market description of DG / UG / TG; Control (abstract) description of DG / UG / TG. Texts verbatim from paper Appendix D / `games_instructions.md` / `stories.md`.
- **Category targets (experience classes):**
  1. moral/sharing problem (dividing something when the other has a claim on your conduct)
  2. cooperation / joint-production problem (acting together to enlarge a pie)
  3. conflict / competition problem (opposed interests over a fixed pie)
  4. market transaction between strangers (prices, anonymity, arm's length)
- **Receiver-type targets:** UG — a counterpart who accepts whatever is profitable vs one who rejects insulting offers out of protest; TG — a counterpart who returns a fair share vs one who keeps.
- **Procedure:** mirrors round 1 — three frontier models × three conversations with independently permuted neutral labels; prompts: (i) 0–100 similarity for each context × category pair; (ii) for each context, 100 retrieval points across the four categories; (iii) for UG/TG contexts, 100 points across the two receiver types ("whom do you picture on the other side?").
- **Hypotheses:** Market description more similar to (4) and (3), less to (1), than Control, in every game; Bonus more similar to (2) than Aid; Aid ≈ Bonus on (1); under Market, receiver-type retrieval shifts toward accepting/returning (consistent with observed P2 behavior and P1 beliefs).
- **Output:** recording workbook in the same format as `LLM_Similarity/memory_games_llm_recording.xlsx`.

## Module B — human survey (Prolific)

- **Sample:** N ≈ 150 raters (no gameplay), ~8–10 min, standard hourly-equivalent pay; attention check + comprehension screen per house standards.
- **Design:** within-subject, randomized order and neutral labels.
  1. Story↔game module: the two stories + the four neutral game descriptions → 0–100 similarity for the 8 pairs, plus the retrieval split (replicates the LLM round 1 with humans).
  2. Context↔category module (from A), assigned via a balanced incomplete design to cap duration.
- **Preregistration (AsPredicted), directional hypotheses:**
  - (a) story↔game similarity ordering DG-KW > DG-LT > UG > TG;
  - (b) |Bonus − Aid| ≈ 0 within game (equivalence bounds ±10 points);
  - (c) Market-vs-Control category shifts as in Module A;
  - (d) Bonus retrieval share increasing in strategic richness (DG-KW → TG).
- **Analysis:** cell means with rank tests across raters; human–LLM rating correlation (the validation statistic for §3.3); category-similarity means feed §2.2's "ideal data" and the model-prediction discussion in §3.3.

## Module C — contrast-weighted similarity (BGLS equation 22; optional, PENDING NG)

*Added 2026-07-13. Written against the pending vignette redesign (NG 07-11): "vignette" below = prototypical situation built from the classification prompts, with well/badly-behaving receiver versions. This module is where the BGLS notation question acquires operational content — adopt their notation in the paper only if this runs.*

- **The object.** BGLS (2026) similarity is feature-by-feature with contrast weights; with linear distance their eq. (22) says attention to feature *i* follows the **description's prominence** σ_δ,i when σ_c,i·F_c/σ_δ,i < 1 and the **category's attention** α_c,i otherwise: the description wins when the category is unfamiliar (low F_c) or the current cue distinctive (σ_δ,i high relative to the category contrast σ_c,i).
- **Feature space (games):** the three payoff elements (own earnings, total surplus, sharing norm) + context features (parties/relationship, commodity at stake, framing vocabulary: transaction/price vs allocation/give). Exact list = call item.
- **Elicitation add-on:** for each context×vignette pair, in addition to the overall 0–100 similarity: (i) prominence of each feature in the context text (σ_δ proxy, 0–100); (ii) how characteristic vs variable each feature is across situations of the vignette's category (σ_c proxy). F_c requires a personal history: humans rate category familiarity ("how often have you experienced situations like this?"); the LLM has no F_c — its module tests only the σ_δ/σ_c structure with F_c held fixed.
- **Aggregation:** contrast-weighted similarity = Σ_i w_i·sim_i with w_i set by the eq.-22 rule (description- or category-driven per the threshold), reported alongside the flat unweighted metric.
- **Testable payoff (what it buys beyond a metric):** (1) the weighted metric should predict the treatment-effect gradient (E5's monotone similarity→effect ranking) at least as well as the flat one; (2) the threshold prediction itself: representations should follow game *structure* on high-σ_cF_c/σ_δ features and follow the *frame* on distinctive unfamiliar cues — the market frame's price/transaction vocabulary is the natural high-σ_δ test case, and the Market-vs-Control classification shifts are the outcome it must predict.
- **Costs:** roughly doubles the per-pair ratings; adds two constructs raters must understand (prominence, typicality); if dropped, the paper's mapping paragraph (§2.2) fully covers the BGLS relation with no measurement claim.
- **Needs NG:** run it or drop it; the feature list; whether F_c-familiarity is worth eliciting on the human side or the test stays within-LLM at fixed F_c.

## Open before fielding (coauthor call)

- Include DG-LT in the human story↔game module (recommended: costs one row, disciplines §5.2).
- Exact category wordings — should they map 1:1 to the classification categories (Moral / Self-interest / Mutual Benefit) or stay in NG's broader triad (moral / cooperation / conflict)? Current draft: broader triad + market as a fourth.
- Whether receiver-type ratings need a separate rater sample to avoid contamination from the sender-side modules.
- Module C (BGLS eq. 22 contrast weights): run it or drop it; feature list; F_c-familiarity elicitation (human side) vs fixed-F_c LLM-only test.

When fielding is approved: draft the AsPredicted with the `/preregistration` skill from this document.
