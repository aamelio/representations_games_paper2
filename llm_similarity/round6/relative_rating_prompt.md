# Relative-similarity rating prompt

You will receive one target decision situation and a set of candidate
vignettes. Read the target and every vignette before assigning any points.

Allocate exactly **1,000 integer points** across the complete set of
vignettes. Give more points to a vignette when the decision situation it
describes is more similar, overall, to the target situation. Zero points are
allowed. Your ratings are relative to the vignettes shown in this task: judge
them jointly, rather than first producing independent absolute scores.

Base your judgment only on the texts. Do not infer anything from the neutral
identifiers, their order, or possible hidden classifications. There is no
required number of vignettes that must receive positive points.

Return JSON only, using this form:

```json
{
  "allocations": [
    {"vignette_id": "V01", "points": 0}
  ],
  "total": 1000
}
```

Include every vignette identifier shown in the task exactly once. Before
answering, verify that all points are nonnegative integers and sum to exactly
1,000.

## Choice sets

- Structural stage: the target is evaluated against all 30 vignettes.
- Within-game stage: the target is evaluated only against the vignettes built
  around the same game structure (6 for DG and 12 each for UG and TG).

The wording of the rating task is otherwise identical across the two stages.
