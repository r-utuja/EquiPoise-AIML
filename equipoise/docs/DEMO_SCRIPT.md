# Demo script (3 minutes)

For presenting Equipoise to examiners. Before you start:

```bash
python scripts/pick_example_patients.py    # once, about 5 min: examples become instant
streamlit run app/streamlit_app.py
```

Open the app once beforehand so the model is loaded (the first start takes about 30 s).
Use a desktop browser at full width.

| Time | Click | Say |
|---|---|---|
| 0:00 | **Home** | "Equipoise helps an eye doctor explain three injection medicines for diabetic macular edema to a patient: Avastin, Lucentis and Eylea. Eylea and Lucentis cost far more than Avastin. The question is: is it worth it for *this* patient?" |
| 0:15 | **Get a suggestion** | "It's a three-step wizard built for a clinician, not an ML person. Vision is entered on the eye chart, there are nine essential fields, and the rest is optional and prefilled." |
| 0:30 | **Try an example → Poorer starting vision**, then **Compare the medicines** | "This is a real held-out test patient, never used for training: poor vision and a thick retina." |
| 0:45 | Point at the headline | "The answer comes in one sentence: *no clear winner*, so Avastin is a reasonable choice. Then how sure we are, in words: *a close call*. Because Eylea's chance of helping by more than a line is about half, we add one honest line: Eylea *may* give a small extra benefit." |
| 1:05 | Point at the cards and chart | "Each card shows lines on the eye chart, the likely range compared with Avastin, and injections. Costs stay hidden until real prices are entered. We never show made-up rupees." |
| 1:20 | **See why** | "What mattered most for this patient: retinal thickness and starting vision. How we decided: Eylea's extra benefit is likely 0.5 to 1.5 lines, which is not *clearly* more than one line. Even with a lower bar of half a line there's still no winner." |
| 1:40 | **Download summary (PDF)** (optional) | "There's a one-page printout for the patient's file." |
| 1:45 | Back to step 1, **Good starting vision**, Compare | "Here the trial and the model agree: no difference, and this time *we're fairly confident*." |
| 2:00 | Sidebar → **Behind the model** | "For you, all the ML is one click away. The glossary maps each term to the plain word on the main pages. The key finding: all 132 held-out patients get *no clear winner*, at both 5 and 3 letters. Protocol T supports the published subgroup result, but it is too small for confident per-patient drug choices, and we report that rather than hide it." |
| 2:25 | Tabs: **Treatment-effect evaluation**, then **Decision logic**, then **Explainability** | "Causal forest chosen on semi-synthetic PEHE. Policy values with bootstrap intervals. The fuzzy layer and the expert-system rule trace for this patient, the Newton-Raphson break-even, and the SHAP waterfall behind 'what mattered most'." |
| 2:45 | **MLOps** tab | "MLflow, DVC, CI and Docker. The UI only *presents* the model's numbers: a presenter layer that is unit-tested, plus a copy test that fails if jargon leaks into the main pages." |
| 3:00 | — | "Questions?" |

## Which example patient to use

| Example | What it shows |
|---|---|
| **Poorer starting vision** | Close call, plus the "Eylea may help a little more" line. This is the most interesting case. |
| **Moderate starting vision** | "We're somewhat confident." |
| **Good starting vision** | "We're fairly confident." The trial finding agrees: no difference at 20/40 or better. |

There is no example where a costlier drug clearly wins, because the model finds none on
the test set. If asked, say so plainly. It is the main finding.

## If something goes wrong

- **Model loading spinner:** wait about 30 s. It's only slow the first time.
- **An example is slow (about 2 s, or about 6 s on "See why"):** the precomputed file is
  missing or stale. Re-run `python scripts/pick_example_patients.py`.
- **To show costs:** open Price check and enter prices for the session. Say clearly that
  they are illustrative.
