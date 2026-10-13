# OCT screening: OpenCV, YOLO classification, ResNet18, Grad-CAM (M10, optional)

> Separate dataset: Kermany et al., *Cell* 2018 (Kaggle `paultimothymooney/kermany2018`).
> These images are **not linked to Protocol T patients**. M10 is a screening front-end
> demo ("does this scan look like DME?"), never an input to the treatment model.

**Data.** The archive is an 11.6 GB zip, but only a small subset is needed. Its cloud
storage supports HTTP range requests, so `scripts/download_oct_subset.py` reads just the
zip's index (central directory) and then fetches only the chosen images: 316 MB instead of
11.6 GB.

**Patient leakage.** File names carry patient IDs, and the official train/test split is
**not patient-disjoint**: in our first draw, 278 test patients also had training scans.
Scans of the same eye look alike, so that would inflate test scores. We therefore:
- keep the whole official test split (242 images per class, 968 in total);
- draw the training subset only from patients with no test scan;
- split validation from training by patient (`GroupShuffleSplit`).

All three pairs of splits now share 0 patients (`m10_patient_overlap.md`). The final sets
are 3,097 training images (1,233 patients) and 903 validation images (309 patients), plus
the 968 test images (609 patients), as listed in `m10_dataset_summary.md`.

**OpenCV preprocessing** (`m10_preprocessing.png`):
- grayscale;
- a **median filter** (removes speckle, keeps edges). **Non-local means** is shown for
  comparison; it is smoother but much slower;
- **CLAHE** (contrast-limited adaptive histogram equalization, tile by tile), which makes
  retinal layers and fluid stand out but also amplifies background speckle;
- resize to 224 × 224.

**Models.** Both get the same folders, image size, 5 epochs and seed, on CPU.
- **ResNet18 transfer learning**: ImageNet weights, a new 4-class head, the backbone frozen
  for epoch 1, then AdamW fine-tuning. The best epoch is chosen on validation.
- **YOLO classification** (`yolo11n-cls`, Ultralytics): a small pretrained classifier,
  fine-tuned.

**What it showed** (`m10_comparison.md`, `m10_per_class.md`, `m10_confusion.png`).
- On the official test set, YOLO reaches **98.8%** accuracy (macro F1 0.988) and ResNet18
  **94.8%** (macro F1 0.948).
- ResNet's main error is calling DRUSEN "CNV" (37 of 242 drusen scans). Both are diseases
  of the deep retinal layers, so they look alike. YOLO makes that mistake 4 times.
- **Validation is harder than test.** The best patient-grouped *validation* accuracy is
  88.7% for ResNet (epoch 4, `m10_resnet_history.md`) and 86.4% for YOLO (epoch 4,
  `m10_yolo_history.md`). The official test set is a curated, expert-graded set and is
  easier than random training scans, so read the test numbers as optimistic for real
  clinics. On validation the two models are close. Five CPU epochs is a short budget;
  more epochs or a GPU would likely help both.

**Grad-CAM** (`m10_gradcam.png`) weights the last convolutional block's feature maps by
the gradient of the predicted class, giving a coarse (7 × 7, upsampled) map of where the
network looked. Most maps centre on the fovea and retinal layers, where these diseases
show, rather than on the image border. That is reassuring, but it is a sanity check, not
proof that the model uses clinically meaningful features.

**Licence note.** Ultralytics YOLO is AGPL-3.0. Its anonymous analytics and its own MLflow
logging are switched off in `vision/yolo.py`.
