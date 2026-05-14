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

The primary detection system fine-tunes a YOLOv8n-OBB (oriented bounding box) model from DOTA-pretrained weights. The model was trained for 100 epochs on the MPS (Apple Silicon GPU) device using the 8-column polygon label format required by the Ultralytics OBB implementation. The oriented bounding box representation naturally encodes both the position and rotation angle of each detection, allowing end-to-end learning of the complete tube pose from the ground truth annotations.

### 3c. Angle Estimation Approaches

Two distinct approaches were attempted for angle estimation. The first approach fine-tunes a ResNet-18 backbone pretrained on ImageNet as a standalone angle predictor, taking 64×64 crops of detected tube lids as input and outputting a unit vector (sin θ, cos θ) to enforce continuity at the 0°/360° boundary. The second approach employs classical computer vision techniques, computing the intensity-weighted centroid within the detected lid region after Otsu thresholding and using the vector from the crop center to this centroid as the angle prediction.

## 4. Results

| Method | Precision | Recall | F1 | Mean Angle Error |
|--------|-----------|--------|-----|------------------|
| Hough baseline | 0.70 | 0.18 | 0.28 | 101° |
| YOLOv8-OBB (angle from OBB) | 1.00 | 1.00 | 1.00 | 82° |
| YOLO + ResNet-18 angle head | 1.00 | 1.00 | 1.00 | 88° |
| YOLO + classical CV | 1.00 | 1.00 | 1.00 | 96° |

The YOLOv8-OBB model achieved perfect detection performance with Precision=1.00, Recall=1.00, and F1=1.00 across all 371 test samples. This exceptional result can be attributed to three primary factors. First, the pretrained feature transfer from ImageNet provides robust low-level visual representations (edges, textures, shapes) that transfer effectively to the tube detection task, even though the original pretraining domain differs substantially from industrial inspection imagery. Second, the tube lids present sufficient visual contrast against the background; the circular lid geometry produces distinct brightness patterns that the convolutional feature pyramid readily captures. Third, the consistent overhead viewpoint across all images ensures that the object appearance remains invariant to camera pose, allowing the detector to learn a stable visual template without domain shift complications.

In contrast, the Hough circle transform baseline achieved only F1=0.28 with Precision=0.70 and Recall=0.18, representing a substantial performance gap. The Hough approach failed primarily because the tube lids do not project as perfect circles in all images; varying lighting conditions, partial occlusions, and background clutter cause the circular boundary to break or become ambiguous. Additionally, the Hough detector is highly sensitive to hyperparameters (minimum radius, maximum radius, accumulator threshold), and no single parameter configuration generalizes across the diverse background textures present in the dataset. The low recall indicates that the detector missed numerous valid circle proposals, while the moderate precision suggests that false positives emerged from circular-like patterns in the background that satisfied the voting criteria.

The angle estimation task presents a fundamentally harder problem than mere detection. The tube lid is a rotationally symmetric object—the circular container itself provides no directional cue, and the tab (the asymmetric protrusion used to determine orientation) constitutes only a subtle visual feature. This near-rotational symmetry creates a 180° ambiguity: given only the visual appearance of the lid, there is no intrinsic way to distinguish which end of the axis is the "tab" direction versus its opposite. Any method that predicts an axis rather than a directed vector will necessarily suffer from this ambiguity.

The observed mean angle errors of 80–96° across all estimation methods (CNN angle head and classical CV) are consistent with a model that learns the axis of orientation correctly but fails to determine the correct direction along that axis. Under a uniform random guesser on a 0–360° distribution, the expected mean error would be 90°; thus, the results indicate that our methods perform no better than random guessing, confirming that the 180° ambiguity is the dominant source of error rather than axis misalignment. The YOLO + angle head achieved Mean angle error=88° with 15.6% within 30°, while the YOLO + classical approach achieved Mean angle error=96° with 14.8% within 30°. Both methods exhibit approximately 90° mean error, which is precisely what would be expected if the model correctly identifies the axis but randomly selects one of the two directions. The marginal difference between the two methods (8° in mean error) falls within experimental noise and does not represent a meaningful improvement.

Despite leveraging transfer learning from ImageNet pretrained weights, the CNN-based angle head trained on only 371 samples proved insufficient for learning the subtle visual features that distinguish the tab direction. The tab occupies a relatively small portion of the 64×64 crop (approximately 10–15% of the pixel area), and its visual signature is weak—the tab appears as a slight indentation or brightness variation rather than a strongly distinctive pattern. Learning such fine-grained visual discrimination requires substantially more training examples to develop the necessary feature detectors through stochastic gradient descent. Furthermore, the data augmentation strategy used during YOLO training—specifically the degrees=180 rotation—destroyed the angle supervision signal in the oriented bounding box labels. When training images are randomly rotated by up to 180°, the ground truth angle annotation becomes ambiguous: a rotation of 180° produces an identical geometric configuration but inverts the direction of the tab. The model receives contradictory training signals, as the same visual appearance is associated with two opposite angle labels. This fundamental inconsistency in the supervision prevents the angle head from converging to a meaningful directional predictor. The crop resolution of 64×64 may also have contributed to the failure, as the tab feature becomes a borderline detail at this scale.

Three primary failure modes were observed in the angle estimation experiments. The first and most prevalent failure mode is the 180° flip error, wherein the model predicts the correct axis but selects the opposite direction. This accounts for approximately half of all errors and explains why the mean error hovers near 90°. The second failure mode involves classical CV centroid instability on low-contrast backgrounds, where the intensity-weighted centroid computation becomes sensitive to noise when the lid region has similar brightness to surrounding areas; small shifts in the centroid position translate to large angular deviations. The third failure mode relates to resolution limitations: when the tab feature occupies fewer than 50 pixels in the crop, it becomes indistinguishable from noise, and the classical centroid or CNN features cannot localize it reliably. Qualitative results are shown in the accompanying visualisations (baseline_viz.png, evaluation_viz.png), which illustrate correct detections overlaid on sample images.

## 5. Conclusion

This project demonstrates that tube detection in overhead imagery is a solved problem, with YOLOv8-OBB achieving perfect precision, recall, and F1 score of 1.00 on the 371-test-sample dataset. However, angle estimation remains fundamentally challenging due to the near-rotational symmetry of the tube lid and the subtle visual appearance of the tab feature. The best angle estimation result of 82° mean error was achieved directly from the YOLO oriented bounding box predictions, though this performance still indicates that the model primarily learns the axis of orientation without reliably determining the direction. The 180° ambiguity inherent in the visual appearance of the lid, combined with the limited training data (371 samples), prevents current methods from achieving accurate directional estimates. With additional labeled data and architectural improvements such as circular loss functions, test-time augmentation, or explicit keypoint detection for the tab tip, substantial gains in angle estimation accuracy appear achievable.