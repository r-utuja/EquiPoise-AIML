# TARNet and the optimizers (SGD, Adam, RMSprop)

**What.** TARNet is a neural network with a shared representation (2 × 64 units) and one
output head per arm (1 × 32 units). The loss is computed only on each patient's
*factual* arm; the other heads give counterfactual predictions. Training uses dropout 0.1,
weight decay and early stopping on a validation fold (patience 25), in PyTorch.

**Optimizers.**
- **SGD with momentum** (0.9) accumulates a velocity of past gradients.
- **RMSprop** divides each step by a running RMS of recent gradients (per-parameter
  learning rates).
- **Adam** combines momentum with RMSprop scaling and adds bias correction.

All three use the same seed, data and budget.

**What it showed** (`m6_tarnet_optimizers.md`, `m6_tarnet_optimizers.png`). All three reach
almost the same best validation loss (0.542 / 0.542 / 0.541, standardised outcome) and
stop early within about 26–32 epochs. RMSprop's best epoch was the first. The network
starts overfitting almost immediately: there is little signal to learn beyond the
arm means. As a CATE learner, TARNet had a worst-case PEHE of 5.28 letters, worse than the
causal forest (`m6_learner_selection.md`).
