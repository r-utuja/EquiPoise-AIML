# Forward-chaining expert system

**What.** A rule engine written from scratch (`decision/expert.py`). Working memory holds
**facts**, for example the benefit interval, the MCID, baseline vision and whether prices
are demo prices. Each **rule** has conditions and an action that adds facts. Each cycle:
1. collect every rule whose conditions hold and which has not fired yet (the agenda);
2. **resolve conflicts**: highest salience first, then the most specific rule (most
   conditions), then the earliest defined;
3. fire it and record a line in the **reasoning trace**.

It repeats until no rule can fire.

**Rules** use only model outputs and trial facts. For example: "benefit CI includes values
below the MCID → equipoise → recommend the cheapest drug"; "baseline vision ≥ 69 → note the
published finding of no advantage". Every rule's docstring states that it is
decision-support logic for the project, **not** clinical guidance.

**What it showed.** The trace is deterministic: the same patient always gives the same
trace, and this is tested. For all 132 held-out patients the engine concludes equipoise and
recommends bevacizumab (`m8_population_summary.md`), because every benefit interval
includes values below 5 letters, and also below the sensitivity threshold of 3 letters.
The trace shows the published < 69-letter rule next to this conclusion: it would give
aflibercept to 50% of these patients.
