# Hidden Markov model, forward algorithm, Viterbi

**What.** An HMM assumes each visit has a hidden disease state that emits what we observe:
here, the change in vision and in CST since the last visit (Gaussian emissions). States
switch according to a transition matrix. We fit a 3-state Gaussian HMM with `hmmlearn` and
label the states by their means.

- The **forward algorithm** computes the likelihood of a sequence by summing over all state
  paths with dynamic programming: αₜ(j) = [Σᵢ αₜ₋₁(i) aᵢⱼ] bⱼ(oₜ), done in log space. We
  implement it from scratch.
- **Viterbi** replaces the sum with a max and backtracks to find the single most likely
  state path for each patient.

**What it showed** (`m7_hmm_checks.md`, `m7_hmm_states.md`). On 519 patients (10,522
visits), our forward log-likelihood matches hmmlearn to 1e-12, and our Viterbi paths are
identical. The states are:
- *improving*: mean CST change −26 µm per visit, mean vision change +1.3 letters; 15% of
  visits, but the most likely start state;
- *active*: CST −5 µm; 45% of visits;
- *stable*: CST −1.6 µm; 40% of visits.

Decoded states are strongly associated with whether an injection was given at that visit
(chi-square p < 0.001, `m7_hmm_state_vs_injection.md`). This is expected, because the
protocol injects while edema is active.
