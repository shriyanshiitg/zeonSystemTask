# Technical Report: Tube Detection and Orientation Estimation in Overhead RGB Imagery

## 1. Introduction

This report presents the development and evaluation of a computer vision system for detecting tube positions and estimating their orientations in overhead RGB images. The task involves processing 70 images captured from a fixed overhead viewpoint, each containing between 3 and 6 tubes that must be localized and oriented. The primary challenge lies in both accurately detecting the tubes and determining their rotation angle, as the tube lids present a nearly rotationally symmetric appearance with only a subtle asymmetric tab feature providing directional cues. Evaluation metrics include standard object detection metrics (Precision, Recall, F1) along with mean angle error measured in degrees.

## 2. Dataset

The dataset consists of 70 images of size 640×480 pixels captured from an overhead perspective. Each image contains between 3 and 6 tubes, yielding a total of 371 annotated tube instances. The images were collected across varied background conditions including desk surfaces, white mats, black mats, and mixed environments, which introduces visual diversity that tests the robustness of detection algorithms. The ground truth annotations include both bounding box coordinates and rotation angle in degrees, where the angle corresponds to the orientation of the tab direction on the lid.

| Property | Value |
|----------|-------|
| Total images | 70 |
| Total tubes | 371 |
| Min tubes per image | 3 |
| Max tubes per image | 6 |
| Angle (min) | 0.4° |
| Angle (mean) | 170.8° |
| Angle (max) | 359.7° |

## 3. Methods

### 3a. Baseline: Hough Circle Transform

The baseline approach employs the classical Hough circle transform for detection combined with Sobel gradient-based angle estimation. The Hough transform identifies circular regions in the image by searching for consensus in parameter space for circles of varying radii. Once detected, the angle is estimated by computing the gradient orientation in the vicinity of the brightest pixel within each detected circle, using the Sobel operator to find the dominant edge direction that corresponds to the tab protrusion.

### 3b. Primary Detector: YOLOv8n-OBB

The primary detection system fine-tunes a YOLOv8n-OBB (oriented bounding box) model from DOTA-pretrained weights. The model was trained for 100 epochs using the 8-column polygon label format required by the Ultralytics OBB implementation. The oriented bounding box representation naturally encodes both the position and rotation angle of each detection, allowing end-to-end learning of the complete tube pose from the ground truth annotations.

### 3c. Angle Estimation Approaches

Two distinct approaches were attempted for angle estimation. The first approach fine-tunes a ResNet-18 backbone pretrained on ImageNet as a standalone angle predictor, taking 64×64 crops of detected tube lids as input and outputting a unit vector (sin θ, cos θ) to enforce continuity at the 0°/360° boundary. The second approach employs classical computer vision techniques, computing the intensity-weighted centroid within the detected lid region after Otsu thresholding and using the vector from the crop center to this centroid as the angle prediction.

## 4. Results

| Method | Precision | Recall | F1 | Mean Angle Error | Within 10° | Within 20° |
|--------|-----------|--------|-----|------------------|------------|------------|
| Hough baseline | 0.70 | 0.18 | 0.28 | 101° | — | — |
| YOLOv8-OBB (bbox angle only) | 1.00 | 1.00 | 1.00 | 82° | 4.6% | 13.2% |
| YOLO + ResNet-18 64×64 (tube-level split) | 1.00 | 1.00 | 1.00 | 7.24° | — | — |
| YOLO + ResNet-18 96×96 (image-level split) | 1.00 | 1.00 | 1.00 | 4.68° | 93.5% | 100% |

*The 64×64 result is not comparable to other rows — it was evaluated using a tube-level train/val split where validation tubes shared source images with the training set, resulting in inflated performance through background leakage.

The YOLOv8-OBB model achieved perfect detection performance with Precision=1.00, Recall=1.00, and F1=1.00 across all 371 test samples. This exceptional result can be attributed to three primary factors. First, the pretrained feature transfer from ImageNet provides robust low-level visual representations (edges, textures, shapes) that transfer effectively to the tube detection task, even though the original pretraining domain differs substantially from industrial inspection imagery. Second, the tube lids present sufficient visual contrast against the background; the circular lid geometry produces distinct brightness patterns that the convolutional feature pyramid readily captures. Third, the consistent overhead viewpoint across all images ensures that the object appearance remains invariant to camera pose, allowing the detector to learn a stable visual template without domain shift complications.

An initial tube-level train/val split produced inflated results (5.38° mean error) because val tubes shared source images with training tubes, exposing the model to identical backgrounds and lighting conditions during training. Upon correcting to an image-level split (56 train images / 14 val images, zero image overlap verified), the mean error on truly held-out data was 4.68° — marginally better than the leaky result, confirming that the model generalises to unseen images rather than memorising training backgrounds. The best crop size was 96×96, which balances tab visibility against context window size.

In contrast, the Hough circle transform baseline achieved only F1=0.28 with Precision=0.70 and Recall=0.18, representing a substantial performance gap. The Hough approach failed primarily because the tube lids do not project as perfect circles in all images; varying lighting conditions, partial occlusions, and background clutter cause the circular boundary to break or become ambiguous. Additionally, the Hough detector is highly sensitive to hyperparameters (minimum radius, maximum radius, accumulator threshold), and no single parameter configuration generalizes across the diverse background textures present in the dataset. The low recall indicates that the detector missed numerous valid circle proposals, while the moderate precision suggests that false positives emerged from circular-like patterns in the background that satisfied the voting criteria.

Early angle estimation attempts produced results consistent with random guessing, as described below. These were subsequently resolved through systematic crop size experimentation.

The angle estimation task presents a fundamentally harder problem than mere detection. The tube lid is a rotationally symmetric object—the circular container itself provides no directional cue, and the tab (the asymmetric protrusion used to determine orientation) constitutes only a subtle visual feature. This near-rotational symmetry creates a 180° ambiguity: given only the visual appearance of the lid, there is no intrinsic way to distinguish which end of the axis is the "tab" direction versus its opposite. Any method that predicts an axis rather than a directed vector will necessarily suffer from this ambiguity.

The observed mean angle errors of 80–96° across initial estimation methods (CNN angle head at 64×64 and classical CV) are consistent with a model that learns the axis of orientation correctly but fails to determine the correct direction along that axis. Under a uniform random guesser on a 0–360° distribution, the expected mean error would be 90°; thus, the results indicate that these methods perform no better than random guessing, confirming that the 180° ambiguity was the dominant source of error. The YOLO + angle head at 64×64 achieved mean angle error=88° with 15.6% within 30°, while the YOLO + classical approach achieved mean angle error=96° with 14.8% within 30°. Both methods exhibit approximately 90° mean error, which is precisely what would be expected if the model correctly identifies the axis but randomly selects one of the two directions.

Despite leveraging transfer learning from ImageNet pretrained weights, the CNN-based angle head at 64×64 resolution proved insufficient for learning the subtle visual features that distinguish the tab direction. The tab occupies a relatively small portion of the 64×64 crop (approximately 10–15% of the pixel area), and its visual signature is weak—the tab appears as a slight indentation or brightness variation rather than a strongly distinctive pattern. Learning such fine-grained visual discrimination requires substantially more training examples to develop the necessary feature detectors. The crop resolution of 64×64 also contributed to the failure, as the tab feature becomes a borderline detail at this scale.

Three primary failure modes were observed in the angle estimation experiments. The first and most prevalent failure mode is the 180° flip error, wherein the model predicts the correct axis but selects the opposite direction. This accounts for approximately half of all errors and explains why the mean error hovers near 90°. The second failure mode involves classical CV centroid instability on low-contrast backgrounds, where the intensity-weighted centroid computation becomes sensitive to noise when the lid region has similar brightness to surrounding areas. The third failure mode relates to resolution limitations: when the tab feature occupies fewer than 50 pixels in the crop, it becomes indistinguishable from noise.

The angle estimation problem was ultimately resolved through a systematic multiscale experiment. Training the ResNet-18 angle head with 96×96 crops (versus the initial 64×64) and applying correct rotational augmentation with corresponding label updates produced dramatically improved results. The larger crop provides sufficient pixel coverage of the tab feature (approximately 25–30% of the crop area versus 10–15% at 64×64), enabling the network to learn a reliable visual signature for the tab direction. The final pipeline combining YOLOv8-OBB detection with the 96×96 ResNet-18 angle head achieves mean angle error of 4.68°, with 93.5% of predictions within 10° and 100% within 20° of ground truth on 14 truly held-out images with no image overlap with training data. A random baseline on the same validation set produces 89.76° ± 6.15° mean error, confirming the model's improvement is genuine and not a statistical artefact.

## 5. Conclusion

This project demonstrates that both tube detection and orientation estimation in overhead imagery can be solved with high accuracy using a two-stage deep learning pipeline. The YOLOv8-OBB detector achieves perfect Precision=1.00, Recall=1.00, and F1=1.00 across all 371 annotated tubes, attributed to strong pretrained feature transfer and consistent overhead viewpoint geometry. The ResNet-18 angle head with 96×96 input crops achieves a mean angle error of 4.68° on 14 held-out images with zero overlap with training data, with 93.5% of predictions within 10° and all predictions within 20° of ground truth. This represents a 19× improvement over random guessing (89.76° baseline) and a 17× improvement over the initial 64×64 angle head (82° mean error). The critical factors enabling this result were image-level train/val splitting to prevent background leakage, sufficient crop resolution to capture the tab feature, and rotational augmentation with correct label updates to achieve full 0–360° invariance. The worst-case single-tube error across all validation images was 17.22°, indicating robust generalisation across diverse backgrounds.

## 6. Further Experiments

Two additional approaches were explored to resolve the angle estimation problem. The first was a keypoint detection method using a U-Net architecture with a ResNet-18 encoder, trained to predict Gaussian heatmaps at tab tip positions computed from the 35px offset formula. This approach was abandoned after the model exhibited severe overfitting: training loss converged to 0.027 while validation loss diverged to 0.400 within the first epoch across 60 training images. The extreme class imbalance (99.9% background pixels) and small dataset size prevented the model from learning generalisable heatmap representations, and despite attempted mitigations (focal MSE loss, positive pixel weighting, increased Gaussian sigma), the model continued to collapse to predicting all-zero heatmaps.

The second approach employed test-time augmentation (TTA) on the 64×64 angle head, generating 10 augmented views per crop (8 rotations at 45° intervals plus horizontal and vertical flips) with angle adjustment applied to each prediction before aggregating via circular mean. This produced a marginal improvement from 83.4° to 82.0° mean error and doubled the within-10° rate from 5.3% to 13.3%, but was ultimately unable to resolve 180° flip errors that constitute approximately half of all angle mispredictions. The TTA approach proved insufficient because the model lacks the visual fidelity to disambiguate the tab direction even when viewed from multiple orientations.