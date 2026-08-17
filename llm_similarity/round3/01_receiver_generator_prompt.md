# Generator prompt — round-3 receiver vignettes (run once, freeze output after review)

Run in a single fresh conversation with the designated generator model (round-2
convention: Claude Fable 5; generator != rater). Record model + date at the top of the
frozen file. Review the output (SN/AA/NG) before any rating conversation starts; after
rating starts, the set is frozen.

Class definitions are keyed to the Player-2 classification scheme of the experiment
(Moral good, Moral bad, Self-interest, Mutual Benefit/Cooperation), reconciled with the
classifier rules as in round 2.

---

***[begin prompt — copy everything between the two rules]***

I am preparing materials for a research project on how people categorize everyday social
and economic situations. I need you to write short vignettes — descriptions of realistic
everyday situations — seen from the side of a person or firm who must RESPOND to what
another party has done: respond to proposed terms, or decide what to do with proceeds
that the other party entrusted or handed over. Please follow the specification exactly.

Write 4 vignettes for each of the following four classes of responders (16 in total):

CLASS G (fair-minded responder): the responder goes along with terms or returns a share
of proceeds because doing so is right, deserved, or evenhanded — they honor what the
other side contributed or was promised, even when keeping more would pay.

CLASS B (indignant responder): the responder reads the other side's conduct as wrong —
one-sided terms, an exploitative arrangement, someone taking advantage — and responds by
refusing the terms or withholding everything, even at a real cost to themselves. The
point of the response is to condemn and punish the conduct, not to gain.

CLASS S (own-payoff responder): the responder decides purely by what leaves them better
off — they go along with any terms that leave them ahead, however lopsided, and when
proceeds are in their hands they keep what they can keep without penalty. The other
side's outcome does not enter the decision.

CLASS C (partnership responder): the responder acts to keep a productive joint
arrangement alive — they respond so that both sides gain and the collaboration can
continue or grow, because the enlarged total is the point of the interaction.

CONSTRAINTS

1. 50-80 words, third person, one named responder as protagonist and one clearly
   identified other party.
2. Within each class: vignettes 1-2 have individual people as responder, vignettes 3-4
   have a firm, shop, or business owner acting in a market role. Vignettes 1 and 3
   involve responding to proposed terms (an offer, a split, a price, an arrangement);
   vignettes 2 and 4 involve deciding what to do with money or goods the other party
   entrusted, advanced, or handed over first. Every class uses this same 2x2 layout so
   that class is not confounded with the character type or the setting.
3. Do NOT use the words "fair", "unfair", "selfish", "generous", "greedy", "cooperate",
   "cooperation", "moral", "trust", "betray", or their derivatives: the class must be
   conveyed by what the responder does and weighs, not by naming it.
4. Do not mention experiments, studies, games, players, laboratories, or questionnaires.
5. Do not use the dollar amounts 6, 12, or 18, and do not use a tripling of any amount.
6. Vary the resources, relationships, and settings across vignettes; vary responder
   names, genders, and ages; do not reuse a name. Do not reuse any protagonist name from
   this list (already used elsewhere in the project): Elena, Malik, Ruth, Jun, Aditya,
   Beatriz, Calvin, Dalia, Farah, Gideon, Hyejin, Isaac, Jocelyn, Karim, Leona, Mateo,
   Noura, Omar, Paloma, Quentin, Rina, Stefan, Talia, Ugo, Valerie, Waleed, Ximena,
   Yusuf, Zora, Anders, Bruna, Chen, Deepa, Emil, Fatima, Gabriel, Helena, Idris,
   Johanna, Kwame, Lidia, Musa, Noemi, Pavel, Rasha, Soren, Tamara, Viktor.
7. The vignette states what the responder does (or is about to do) and what drives it —
   but conveys the driver through the situation and the responder's reasoning, never by
   labeling their character.

OUTPUT FORMAT

Return a single JSON object, no other text:
{"receiver_vignettes": [{"id": "<G1..G4|B1..B4|S1..S4|C1..C4>",
"class": "<fair_minded|indignant|own_payoff|partnership>",
"character": "<individual|firm>", "setting": "<proposed_terms|entrusted_proceeds>",
"text": "<the vignette>"}, ...]}

***[end prompt]***
