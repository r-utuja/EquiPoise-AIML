"""Every user-facing string on the main-flow pages (Home, Get a suggestion, Price check,
Progress check, About the trial).

Written for an eye doctor or counsellor talking a patient through options: short sentences,
eye-chart lines rather than letters, brand names first. `tests/test_copy.py` fails if any
technical term from the banned list appears here or in the main-flow page files. The
"Behind the model" and OCT pages keep their own technical wording.
"""

from __future__ import annotations

# ---------------------------------------------------------------- app-wide
APP_NAME = "Equipoise"
NOTICE = "Research prototype, not for clinical use."
DEMO_MODEL = "Demo model: trained on made-up data, not on the trial."
NAV_HOME = "Home"
NAV_SUGGEST = "Get a suggestion"
NAV_PRICE = "Price check"
NAV_PROGRESS = "Progress check"
NAV_ABOUT = "About the trial"
NAV_SECTION_MAIN = "Equipoise"
NAV_SECTION_REVIEW = "For reviewers"
NAV_BEHIND = "Behind the model"
NAV_OCT = "OCT screening (demo)"
ERROR_MODEL = ("We couldn't load the comparison model. Ask your team to run the setup steps "
               "in the README, then reload this page.")
ERROR_REJECTED = "We couldn't use these details. Please check them and try again."
NEED_PATIENT = "Enter a patient first, then come back here."
BTN_GO_STEP1 = "Enter patient details"
LOADING_MODEL = "Loading the comparison model. The first start takes about 30 seconds…"
NOT_KNOWN = "Not known"
YES = "Yes"
NO = "No"

DRUG_BRAND = {"Bevacizumab": "Avastin", "Ranibizumab": "Lucentis", "Aflibercept": "Eylea"}
DRUG_GENERIC = {"Bevacizumab": "bevacizumab", "Ranibizumab": "ranibizumab",
                "Aflibercept": "aflibercept"}
DRUG_FULL = "{brand} ({generic})"

LINE = "line"
LINES = "lines"
LETTERS_APPROX = "≈ {n} letters"

# ---------------------------------------------------------------- home
HOME_TAGLINE = ("Compare three eye-injection medicines for diabetic macular edema, see which "
                "is likely to work better for this patient, and what each would cost.")
HOW_IT_WORKS = "How it works"
HOME_STEPS = (
    (":material/edit_note:", "Enter patient details",
     "Vision, retinal thickness and a few health details."),
    (":material/compare_arrows:", "See the comparison",
     "How much each medicine is likely to help after one year."),
    (":material/lightbulb:", "Understand why",
     "What mattered most, and whether the extra cost is worth it."),
)
HOME_CTA = "Get a suggestion"
HOME_EXAMPLE_LINK = "Try an example patient"
HOME_ABOUT_LINK = "About the trial"

# ---------------------------------------------------------------- wizard
WIZARD_TITLE = "Get a suggestion"
STEP_LABEL = "Step {n} of 3 · {name}"
STEP_NAMES = ("Patient details", "The comparison", "Why this result")
TIP_TITLE = "How to use this"
TIP_LINES = (
    "1. Pick the patient's vision and fill in what you know. Leave the rest.",
    "2. Press **Compare the medicines** to see how the three stack up.",
    "3. Press **See why** to understand the result.",
)
TIP_DISMISS = "Got it"

EXAMPLES_LABEL = "Try an example"
EXAMPLE_NAMES = {"poorer": "Poorer starting vision", "moderate": "Moderate starting vision",
                 "good": "Good starting vision"}
EXAMPLE_HELP = {
    "poorer": "A real patient from the trial's test group with poor vision and a thick retina.",
    "moderate": "A real patient from the trial's test group with moderate vision.",
    "good": "A real patient from the trial's test group with good vision and a thin retina.",
}
EXAMPLE_LOADED = "Loaded the example: {name}. You can change any detail before comparing."
EXAMPLES_UNAVAILABLE = "Examples aren't available here."

ESSENTIALS_TITLE = "Patient details"
VISION_LABEL = "Vision in the treated eye today"
VISION_HELP = ("Best-corrected reading on the eye chart. The trial included people between "
               "20/32 and 20/320.")
VISION_OPTION = "{feet} ({metric})"
EXACT_TOGGLE = "Enter exact letter score"
EXACT_TOGGLE_HELP = "For clinics that record the number of letters read on the chart."
EXACT_LABEL = "Letter score (24 to 78)"
EXACT_HELP = "Number of letters read correctly. 85 letters is about 20/20."
EXACT_NEAREST = "On the eye chart, that is about {vision}."
SEX_OPTIONS = ("Male", "Female")

MORE_TITLE = "More details (optional)"
MORE_NOTE = "We've used typical values for anything you leave blank."
MORE_GROUPS = ("Eye", "Past eye treatment", "General health")

BTN_COMPARE = "Compare the medicines"
BTN_START_OVER = "Start over"
BTN_EDIT = "Edit details"
BTN_SEE_WHY = "See why"
BTN_BACK_COMPARISON = "Back to the comparison"

SPINNER_COMPARE = "Comparing the three medicines…"
SPINNER_WHY = "Working out what mattered most…"
SPINNER_PRICE = "Checking the prices…"
SPINNER_PROGRESS = "Working out the likely path…"

VALIDATION_TITLE = "Please check these details:"
VALIDATION_RANGE = "{label} should be between {lo} and {hi}{unit}."
VALIDATION_BP = "The top blood-pressure number should be higher than the bottom number."

# Field labels, one-sentence help, unit, and the words used when a detail is above or below
# the typical value ("A thicker retina made Eylea's advantage a little larger").
FIELDS: dict[str, dict[str, str]] = {
    "X_va_letters": {"label": "Vision in the treated eye", "help": VISION_HELP,
                     "more": "Better starting vision", "less": "Poorer starting vision"},
    "X_cst_um": {"label": "Retinal thickness from OCT report", "unit": "µm",
                 "help": "Thickness at the centre of the retina, as printed on the OCT report.",
                 "more": "A thicker retina", "less": "A thinner retina"},
    "X_oct_machine": {"label": "OCT machine",
                      "help": "Machines measure thickness differently, so we need to know "
                              "which one was used."},
    "X_hba1c": {"label": "HbA1c", "unit": "%",
                "help": "Most recent HbA1c, a measure of blood sugar over the last 3 months.",
                "more": "Higher HbA1c", "less": "Lower HbA1c"},
    "X_diabetes_duration_yrs": {"label": "Years with diabetes",
                                "help": "How long ago diabetes was diagnosed.",
                                "more": "Longer diabetes", "less": "Shorter diabetes"},
    "X_female": {"label": "Sex", "help": "As recorded in the patient's file."},
    "X_va_below_69": {"label": "Starting vision of 20/50 or worse", "help": ""},
    "X_age": {"label": "Age", "unit": "years", "help": "Age today.",
              "more": "Older age", "less": "Younger age"},
    "X_sbp": {"label": "Blood pressure, top number", "unit": "mmHg",
              "help": "Systolic blood pressure, measured today or recently.",
              "more": "Higher blood pressure", "less": "Lower blood pressure"},
    "X_dbp": {"label": "Blood pressure, bottom number", "unit": "mmHg",
              "help": "Diastolic blood pressure, measured today or recently.",
              "more": "Higher blood pressure", "less": "Lower blood pressure"},
    # optional: eye
    "X_fellow_va_letters": {"label": "Vision in the other eye (letter score)",
                            "help": "Letters read with the other eye. 85 letters is about 20/20.",
                            "more": "Better vision in the other eye",
                            "less": "Poorer vision in the other eye"},
    "X_subretinal_fluid": {"label": "Fluid under the retina at the centre",
                           "help": "Seen on the OCT scan."},
    "X_epiretinal_membrane": {"label": "Epiretinal membrane", "help": "Seen on the OCT scan."},
    "X_iop": {"label": "Eye pressure", "unit": "mmHg", "help": "Pressure in the treated eye.",
              "more": "Higher eye pressure", "less": "Lower eye pressure"},
    "X_pseudophakic": {"label": "Lens replaced (cataract surgery done)",
                       "help": "Whether the treated eye has had cataract surgery."},
    "X_dr_severity_clinical": {"label": "Diabetic retinopathy stage (clinic exam)",
                               "help": "Stage of diabetic eye disease from the clinic exam."},
    "X_dr_severity_rc": {"label": "Diabetic retinopathy grade from photos (1 to 10)",
                         "help": "Grade from retinal photographs, from 1 (none) to 10 (most "
                                 "severe).",
                         "more": "A higher retinopathy grade",
                         "less": "A lower retinopathy grade"},
    # optional: past eye treatment
    "X_prior_dme_trt": {"label": "Any earlier treatment for macular edema in this eye",
                        "help": "Injections, laser or other treatment for macular edema."},
    "X_prior_antivegf": {"label": "Earlier eye injections for macular edema in this eye",
                         "help": "Any earlier Avastin, Lucentis, Eylea or similar injection."},
    "X_prior_focal_laser": {"label": "Earlier focal or grid laser in this eye",
                            "help": "Laser aimed at the macula."},
    "X_prior_prp": {"label": "Earlier scatter laser (PRP) in this eye",
                    "help": "Laser over the wider retina for proliferative retinopathy."},
    # optional: general health
    "X_log_uacr": {"label": "Kidney test: urine albumin-to-creatinine ratio", "unit": "mg/g",
                   "help": "From a urine test, in mg/g as recorded in the trial.",
                   "more": "A higher kidney-test result", "less": "A lower kidney-test result"},
    "X_bmi": {"label": "Body-mass index (BMI)", "unit": "kg/m²",
              "help": "Weight in kg divided by height in metres squared.",
              "more": "A higher BMI", "less": "A lower BMI"},
    "X_insulin": {"label": "Uses insulin", "help": "Currently takes insulin."},
    "X_diabetes_type": {"label": "Diabetes type", "help": "Type 1 or type 2 diabetes."},
    "X_hypertension": {"label": "High blood pressure diagnosed",
                       "help": "Has been told they have high blood pressure."},
    "X_cad": {"label": "Heart disease (coronary)", "help": "Known coronary artery disease."},
    "X_prior_mi": {"label": "Earlier heart attack", "help": "Has had a heart attack."},
    "X_prior_stroke": {"label": "Earlier stroke", "help": "Has had a stroke."},
    "X_renal_disease": {"label": "Known kidney disease", "help": "Has been told they have "
                                                                "kidney disease."},
    "X_smoker_current": {"label": "Smokes now", "help": "Currently smokes."},
    "X_race": {"label": "Race (as recorded in the trial)",
               "help": "The trial recorded these groups; leave as is if unsure."},
    "X_hispanic": {"label": "Hispanic or Latino", "help": "As recorded in the trial."},
}
OPTIONAL_GROUPS: dict[str, tuple[str, ...]] = {
    "Eye": ("X_fellow_va_letters", "X_subretinal_fluid", "X_epiretinal_membrane", "X_iop",
            "X_pseudophakic", "X_dr_severity_clinical", "X_dr_severity_rc"),
    "Past eye treatment": ("X_prior_dme_trt", "X_prior_antivegf", "X_prior_focal_laser",
                           "X_prior_prp"),
    "General health": ("X_log_uacr", "X_bmi", "X_insulin", "X_diabetes_type",
                       "X_hypertension", "X_cad", "X_prior_mi", "X_prior_stroke",
                       "X_renal_disease", "X_smoker_current", "X_race", "X_hispanic"),
}

# ---------------------------------------------------------------- step 2: the comparison
HEAD_NO_WINNER = ("No clear winner for this patient. All three are likely to work about "
                  "equally well, so the lower-cost option, {ref_full}, is a reasonable choice.")
HEAD_WINNER = ("{drug_full} is likely to help this patient most — about {gain} more on the "
               "eye chart than {ref}.")
HEAD_DISCUSS = ("{drug_full} is likely to help a little more, but at these prices the extra "
                "cost may not be worth it. Talk it through with the patient.")
MAY_BENEFIT = ("{drug} may give a small extra benefit for this patient — worth discussing if "
               "cost is not a concern.")
CONFIDENCE = {"fairly": "We're fairly confident.", "somewhat": "We're somewhat confident.",
              "close": "This is a close call."}
TRIAL_FINDING_POOR = ("In the original trial, Eylea helped more on average for people whose "
                      "vision started at 20/50 or worse, like this patient.")
TRIAL_FINDING_GOOD = ("In the original trial, people whose vision started at 20/40 or better "
                      "did about equally well on all three.")
UNCERTAINTY = "These are estimates from one trial of 660 people. Treat them as a rough guide."

TAG_SUGGESTED = "Suggested"
TAG_LOWEST_COST = "Lowest cost"
CARD_IMPROVE_LABEL = "Expected improvement after 1 year"
GAIN = "About {n} {unit}"
LOSS = "About {n} {unit} worse"
LITTLE_CHANGE = "Little change"
CARD_REFERENCE_NOTE = "The other two are compared with Avastin."
RANGE_PREFIX = "Compared with {ref}: "
RANGE_MORE = "likely between {lo} and {hi} more {unit}"
RANGE_ABOUT = "likely about {n} more {unit}"
RANGE_FEWER = "likely between {lo} and {hi} fewer {unit}"
RANGE_MIXED = "likely between {lo} fewer and {hi} more {unit}"
RANGE_NONE = "likely about the same"
CARD_INJECTIONS = "About {n} injections in the first year"
CARD_COST = "About {amount} in the first year"
COSTS_HIDDEN_NOTE = "Costs appear once current drug prices are added."

CHART_AXIS = "Lines gained on the eye chart after 1 year"
CHART_SUMMARY_DRUG = "{drug}: {gain}."
CHART_SUMMARY_BANDS = ("The shaded bands show how much more {drugs} are likely to help than "
                       "{ref}.")
CHART_SUMMARY_NONE_AHEAD = "None of the medicines is clearly ahead."
CHART_SUMMARY_AHEAD = "{drug} is clearly ahead."
AND = "and"
TO = "to"

# ---------------------------------------------------------------- step 3: why
WHAT_MATTERED_TITLE = "What mattered most"
WHAT_MATTERED_INTRO = "For {drug} compared with {ref}, these details made the most difference:"
FACTOR = "{subject} made {drug}'s advantage {size} {direction}."
FACTOR_SIZE = {"small": "slightly", "medium": "a little", "large": "noticeably"}
FACTOR_LARGER = "larger"
FACTOR_SMALLER = "smaller"
SUBJECT_VALUE = "{phrase} ({value})"
SUBJECT_TYPICAL = "A typical value for {label}"
SUBJECT_IS = "{label}: {value}"
WHAT_MATTERED_UNAVAILABLE = "This explanation isn't available for the current model."

HOW_DECIDED_TITLE = "How we decided"
DECIDE_ESTIMATE = ("We estimated how much each medicine is likely to help this patient after "
                   "one year.")
DECIDE_NOT_CLEAR = "{drug} compared with {ref}: {range}. That's not clearly more than 1 line."
DECIDE_CLEAR = "{drug} compared with {ref}: {range}. That's clearly more than 1 line."
DECIDE_LOWER_BAR_SAME = "Even with a lower bar of about {bar}, there's still no clear winner."
DECIDE_LOWER_BAR_DIFF = ("With a lower bar of about {bar}, at least one of the costlier "
                         "medicines would count as clearly better.")
DECIDE_WORTH = "At these prices, the extra benefit is worth the extra cost."
DECIDE_NOT_WORTH = "At these prices, the extra benefit isn't worth the extra cost."
HALF_LINE = "half a line"

STRENGTH = {"weak": "Weak case for paying more", "moderate": "Moderate case for paying more",
            "strong": "Strong case for paying more"}
STRENGTH_LINE = "{strength} for {drug}."
STRENGTH_HIDDEN = "The case for paying more appears once current drug prices are added."

WORTH_TITLE = "Is the extra cost worth it?"
COMPARE_WITH = "Compare {ref} with"
PRICE_SLIDER = "Price of one {drug} injection (₹)"
VALUE_LINE_LABEL = "How much is one extra line of vision worth to the patient? (₹)"
VALUE_LINE_HELP = ("Your judgement, made with the patient. We use it to weigh better vision "
                   "against extra cost.")
BE_SENTENCE = ("If one {drug} injection costs more than about {price}, it stops being worth "
               "the extra cost for this patient.")
BE_AT_PRICE_WORTH = "At {price} per injection, {drug} is worth the extra cost."
BE_AT_PRICE_NOT = "At {price} per injection, {drug} isn't worth the extra cost."
BE_NEVER = ("Even at a very low price, {drug} isn't worth the extra cost for this patient, "
            "with the value you set for one line.")
BE_ALWAYS = ("Up to {price} per injection, {drug} is worth the extra cost for this "
             "patient.")
BE_NEED_VALUE = "Enter how much one extra line of vision is worth to see the answer."
BE_GO_PRICES = "Add prices on the Price check page"

PDF_BUTTON = "Download summary (PDF)"
PDF_FILE = "equipoise_patient_summary.pdf"
PDF_TITLE = "Equipoise - patient summary"
PDF_PATIENT = "Patient"
PDF_PATIENT_LINE = "Vision today about {vision}. Retinal thickness {cst}."
PDF_RESULT = "Result"
PDF_MEDICINES = "The three medicines"
PDF_WHY = "What mattered most"
PDF_FOOTER = ("Research prototype for a college project, not for clinical use. It is not a "
              "medical device and gives no clinical advice.")

# ---------------------------------------------------------------- price check
PRICE_TITLE = "Price check"
PRICE_INTRO = "Is a costlier medicine worth it for this patient? Move the price and see."
SESSION_PRICES_TITLE = "Enter current prices for this session"
SESSION_PRICES_HELP = ("These stay in this browser session only. To set them for everyone, "
                       "ask your team to add prices to the settings file.")
PRICE_FIELD = "{drug} price per injection (₹)"
LASER_FIELD = "Laser session (₹)"
BTN_USE_PRICES = "Use these prices"
BTN_CLEAR_PRICES = "Clear these prices"
PRICES_SESSION = "Using prices entered in this session."
PRICES_TEAM = "Using your team's prices."
PRICES_INVALID = "Please enter a price above zero for each medicine."

# ---------------------------------------------------------------- progress check
PROGRESS_TITLE = "Progress check"
PROGRESS_INTRO = ("This answers a different question from the suggestion: how is a patient "
                  "on treatment doing so far? It doesn't change the suggestion.")
PROGRESS_DRUG = "Medicine the patient is on"
PROGRESS_START = "Vision at the start of treatment"
PROGRESS_START_FROM = "Vision at the start, from the patient details: about {vision}."
VISIT_LABELS = ("Vision at the 1-month visit", "Vision at the 2-month visit",
                "Vision at the 3-month visit")
PROGRESS_EXACT = "Enter exact letter scores"
PROGRESS_BTN = "Show expected progress"
PROGRESS_RESULT = "On this path, expected vision at 1 year is about {vision} ({letters})."
PROGRESS_BETTER = "That's about {gain} better than at the start."
PROGRESS_WORSE = "That's about {gain} worse than at the start."
PROGRESS_SAME = "That's about the same as at the start."
PROGRESS_ERROR = "This could easily be about {gain} better or worse."
PROGRESS_AXIS = "Lines gained since the start"
PROGRESS_TIMES = ("Start", "1 month", "2 months", "3 months", "1 year")
PROGRESS_SUMMARY = ("Vision went from about {start} at the start to about {last} at 3 months. "
                    "The dashed line shows the expected path to 1 year.")
PROGRESS_UNAVAILABLE = "The progress check isn't available for this model."

# ---------------------------------------------------------------- about the trial
ABOUT_TITLE = "About the trial"
ABOUT_WHAT_TITLE = "What the trial was"
ABOUT_WHAT = ("The suggestions come from a large study called Protocol T, run by the DRCR "
              "Retina Network. It included 660 adults with diabetic macular edema at 88 clinics "
              "in the United States in 2012 and 2013. Each person was given Avastin, Lucentis "
              "or Eylea and followed for 2 years.")
ABOUT_FAIR_TITLE = "Why this lets us compare the medicines fairly"
ABOUT_FAIR = ("Each person's medicine was picked by chance, like tossing a coin. Because the "
              "coin doesn't care how sick anyone is, the three groups started out alike, so "
              "differences later come from the medicine.")
ABOUT_FOUND_TITLE = "What the trial found"
ABOUT_FOUND_ALL = ("After 1 year, people gained about {afl} with Eylea, {ran} with Lucentis "
                   "and {bev} with Avastin, on average.")
ABOUT_FOUND_POOR = ("For people whose vision started at 20/50 or worse, Eylea did better: "
                    "about {afl}, against {ran} with Lucentis and {bev} with Avastin.")
ABOUT_FOUND_GOOD = ("For people whose vision started at 20/40 or better, all three did "
                    "about the same: {afl} with Eylea, {ran} with Lucentis and {bev} with "
                    "Avastin.")
ABOUT_FOUND_INJ = "Most people needed {inj} injections in the first year."
ABOUT_FOUND_MISSING = "The trial results haven't been built on this computer yet."
ABOUT_LIMITS_TITLE = "What this app can and can't tell you"
ABOUT_LIMITS = (
    "It learns from one trial of 660 people in the United States, treated in 2012 and 2013.",
    "Differences between individual patients are hard to pin down with this many people, so "
    "most results say there is no clear winner.",
    "Lucentis was given at a lower dose (0.3 mg) than is common in India (0.5 mg).",
    "Prices are entered by your team, not taken from the market. It is not medical advice.",
)
