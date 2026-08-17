# Fable-rater refusal record (round 2)

Appendix H states the third Fable-5 conversation completed only after the
model's safety filter declined the benign opening message **eleven times
across eight candidate label orders**. Refused conversations produce no
transcript, so the contemporaneous record lives in the runner's inline
comments (extracted verbatim below, from `03_api_runner.py`) and in the
seed-by-seed narrative of `ng_comments_tracker.md` (2026-07-19 entries),
which is the authoritative tally.

## Eight candidate label orders for the third conversation

Sets 3, 4 (seed 20260719), 5 (seed 20260720), then the pre-committed
successor-seed probe hunt over 20260721-24, and 20260725 (= set 6, the
order that completed).

## Inline run record from 03_api_runner.py (verbatim)

```
    # set 4: substitute permutation for the Fable-5 robustness rater's third conversation
    # (random.Random(20260719), drawn 2026-07-19): set 3's opening message was declined three
    # times by that model's safety filter (false positive, order-dependent); permutations are
    # nuisance parameters, so a fresh draw preserves the protocol. Opus headline = sets 1-3.
    # set 5: second substitute for the same conversation (random.Random(20260720), successor
    # seed, identical draw procedure — verified to reproduce set 4 from 20260719). Drawn
    # 2026-07-19 after set 4's opening message was also declined (5th refusal) while a same-day
    # probe of set 1's opening message completed cleanly — i.e., the filter's false positive is
    # ordering-specific, not account state. Selection caveat: the third conversation's
    # permutation is thereby conditioned on passing the filter; disclosed in the appendix.
    # set 6: the substitute permutation that completed the third Fable-5 conversation
    # (2026-07-19). Found by a pre-committed successor-seed hunt (probe = opening message only,
    # one call per seed, stop at first acceptance): set 5 (20260720) refused; 20260721-24
    # refused; 20260725 accepted. The filter then refused the IDENTICAL accepted message on the
    # first full-run attempt and accepted it on the next, so its false positives are stochastic
    # per request, not a deterministic function of the ordering — which makes retrying a fixed
    # permutation selection-free. Full run completed on retry attempt 1 after cleanup; refusal
    # tally and the disclosure are in the appendix and ng_comments_tracker.md.
```

## Completed-run transcripts present in this module

- rater_claude_fable_set1.json
- rater_claude_fable_set2.json
- rater_claude_fable_set6.json
- rater_claude_set1.json
- rater_claude_set2.json
- rater_claude_set3.json

Fable conversations completed: 3 (sets 1, 2, 6) — matching the
appendix's statement that all three permuted conversations completed, the
third on the set-6 substitute order.
