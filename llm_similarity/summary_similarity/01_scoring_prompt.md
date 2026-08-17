# Summary-similarity scoring — participants' story summaries vs. categories (task 7)

**Status: DRAFT (SN + Claude, 2026-07-29), for AA/NG review before the production run.**
Origin: NG's post-meeting proposal — score each participant's own summary of the
Aid/Bonus story against the representation categories. Two uses: (1) validation — if Aid
summaries are in aggregate closer to Self-interest than Bonus summaries, the category
salience shift is established in participants' minds, not only in the story texts as
written (round-2 result: Aid +2.5 toward Self-interest exemplars, Bonus +2.1 toward
Mutual Benefit); (2) heterogeneity — within a story condition, the summary provides a
pre-game, pre-decision, person-level measure of which category the story activated,
usable to predict reasons, actions, and beliefs (AA's caveat: faithful-summary
instructions may compress within-story variance; the pilot's variance audit gates this
use).

**Design principles.** The rater sees ONE summary and nothing else — no story name, no
game, no condition, no other summaries' scores. Scores are graded 0–100 per category
(not a forced classification), because within a story condition all summaries share the
same facts: variation in the scores is variation in what the writer foregrounds.
Scoring the text as-is is correct for both uses: between-story differences are the
validation object, and within-story variance (base content constant) is the emphasis
heterogeneity. Category definitions mirror the classifier's Player-1 scheme, worded for
narrative emphasis rather than for a decision reason.

---

## Scoring prompt (one summary per call; batched variant for production TBD)

***[begin prompt]***

I am running a research project on how people retell short stories about everyday
economic situations. Below is one person's written retelling of a story. Read it and
rate, on a 0-100 scale each, how strongly the retelling emphasizes each of the following
three aspects of the situation. The three ratings are independent: they need not sum to
any total, and a retelling can score high on more than one aspect or low on all three.

ASPECT F (fairness and desert): what would be right or deserved; another party's claim,
need, or contribution; whether and how something should be divided or shared.

ASPECT O (own position): the main character's own needs, hardship, or scarcity; what
they get to keep, use, or protect for themselves; their own material situation.

ASPECT J (joint outcomes and relationships): working together, joint contribution or
joint success; an ongoing relationship between the parties; benefits that arise for both
sides from the interaction.

Base your ratings only on what this retelling includes, foregrounds, or dwells on -- not
on what you imagine the full story might contain.

OUTPUT FORMAT: return a single JSON object, no other text:
{"F": <0-100>, "O": <0-100>, "J": <0-100>,
"dominant": "<F|O|J|balanced>",
"note": "<one short phrase naming what the retelling foregrounds>"}

RETELLING:

{summary text}

***[end prompt]***

---

## Protocol notes

- **Mapping** (applied ex post, never shown to the rater): F -> Moral, O ->
  Self-interest, J -> Mutual Benefit/Cooperation.
- **Sample**: all Player-1 story-condition participants (Bonus and Aid, four games);
  Player-2 extension optional later.
- **Pilot** (before any production run): ~50 summaries stratified by story x game;
  outputs: score distributions and within-story variance (the gate for use 2),
  Aid-Bonus contrast direction (sanity for use 1), and a read of the `note` fields
  against the texts.
- **Model**: pilot on the available Anthropic key; production model to be decided with
  AA/NG (the reasons classifier used GPT-4.1 -- family consistency vs. availability;
  disclose either way). Human-coded validation subsample per the existing protocol if
  this becomes a paper exhibit.
- **Blinding**: summaries are submitted in randomized order with neutral ids; the
  scoring script never passes condition, game, or any behavioral variable.
