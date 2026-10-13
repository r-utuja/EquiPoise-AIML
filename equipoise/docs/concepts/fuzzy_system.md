# Fuzzy inference system

**What.** Fuzzy logic lets a value be partly "small" and partly "large" at the same time,
through membership functions. A Mamdani system (`scikit-fuzzy`) does four things:
1. **fuzzifies** each input,
2. applies readable IF–THEN rules (AND = min, OR = max),
3. **aggregates** the clipped output sets,
4. **defuzzifies** by the **centroid** of the aggregated shape.

**Here.**
- Inputs: predicted benefit vs the cheapest drug (letters), certainty = P(benefit > MCID),
  extra cost (share of the cost that is extra), and difference in injection burden.
- Output: a recommendation strength from 0 to 100 for the more expensive drug.
- The rule base has 10 rules (`m8_fuzzy_rules.md`). Membership functions are
  plotted in `m8_fuzzy_membership.png`, with a worked example in
  `m8_fuzzy_worked_example.png`.

Tests check that the output always stays within [0, 100] and never decreases as the
benefit increases.

**What it showed.** For the typical training patient, the strength for aflibercept is 17
out of 100 (`m8_decision.json`). The benefit is a few letters, but the certainty of
exceeding the MCID is near zero and the demo extra cost is large, so the fuzzy layer
leans clearly toward the cheaper drug, consistent with the equipoise call.
