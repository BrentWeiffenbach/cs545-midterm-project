# Mid-Term Project: Night-Time Image Enhancement

## Goal

Given a pair of images of the same scene—one captured at night and the other at day—design and implement your best method to enhance the night-time image so that it matches the day-time appearance as closely as possible.

---

## Evaluation Metrics

### Primary Metric: Mean Squared Error (MSE)

**Mean Squared Error (MSE)** measures the average squared difference between corresponding pixels of two images. If your enhanced image is compared to a reference day-time image, the formula for MSE is:

$$MSE = \frac{1}{N} \sum (\hat{y} - y)^2$$

**Where:**

* $\hat{y}$: Pixel value in the enhanced image
* $y$: Pixel value in the day-time image
* $N$: Total number of pixels (or pixel values)

### Channel-wise MSE

Images typically have 3 color channels: **Red (R)**, **Green (G)**, and **Blue (B)**. Instead of computing one MSE over all channels combined, channel-wise MSE computes MSE separately for each channel:

$$MSE_R = \frac{1}{N} \sum (R_{enhanced} - R_{day})^2$$
$$MSE_G = \frac{1}{N} \sum (G_{enhanced} - G_{day})^2$$
$$MSE_B = \frac{1}{N} \sum (B_{enhanced} - B_{day})^2$$

Then you can average them as the final difference:

$$MSE_{overall} = \frac{MSE_R + MSE_G + MSE_B}{3}$$

> **Note:** Please report this final averaged number in your report and slides.

---

## Important Note on Method Choice

Students are allowed to use any method to achieve the goal—classical, learning-based, or hybrid approaches.
If your chosen method requires training or fine-tuning, you must prepare the training data yourself.
 The provided image pair is not sufficient for robust training, so you
are expected to design or collect additional data (e.g., augmentations,
synthetic generation, or external datasets) as needed.
Ensure that your pipeline remains generalizable and does not overfit to the single provided pair.

## Dataset

You are provided with one pair of images of the same scene (please check the files in Files/Mid-Term/):

* night.png (or .jpg): Night-time capture.
* day.png (or .jpg): Day-time capture (the reference/target).

Both images may differ in exposure, noise, white balance, and viewpoint (small shifts).
Optional: If we release additional hidden pairs,
your method will be re-run on those for final grading. Design your
pipeline to be robust and not overfit to a single visible pair.
Important constraint (integrity rule): During inference, your pipeline must use only the night-time image. The day-time image can be used for training/supervision, parameter selection, or histogram matching, but it must not be directly copied or blended at test time.

## Academic Integrity & Collaboration

You may discuss high-level ideas, but all code must be your own or clearly credited (for any third-party snippets/models).
Cite all sources and pretrained models.
Clearly mark any borrowed components in the report and code headers.
