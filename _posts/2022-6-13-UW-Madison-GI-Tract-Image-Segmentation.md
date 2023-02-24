---
layout: post
tittle: UW-Madison GI Tract Image Segmentation
category: blog
---
![header](https://raw.githubusercontent.com/tkhangdl/tkhangdl.github.io/master/images/uwgi/header.png)
## 1. Introduction: <a name="introduction"></a>
In 2019, an estimated 5 million people were diagnosed with a cancer of the gastro-intestinal tract worldwide. Of these patients, about half are eligible for radiation therapy, usually delivered over 10-15 minutes a day for 1-6 weeks. Radiation oncologists try to deliver high doses of radiation using X-ray beams pointed to tumors while avoiding the stomach and intestines. With newer technology such as integrated magnetic resonance imaging and linear accelerator systems, also known as MR-Linacs, oncologists are able to visualize the daily position of the tumor and intestines, which can vary day to day. In these scans, radiation oncologists must manually outline the position of the stomach and intestines in order to adjust the direction of the x-ray beams to increase the dose delivery to the tumor and avoid the stomach and intestines. This is a time-consuming and labor intensive process that can prolong treatments from 15 minutes a day to an hour a day, which can be difficult for patients to tolerate—unless deep learning could help automate the segmentation process. A method to segment the stomach and intestines would make treatments much faster and would allow more patients to get more effective treatment.

The UW-Madison Carbone Cancer Center is a pioneer in MR-Linac based radiotherapy, and has treated patients with MRI guided radiotherapy based on their daily anatomy since 2015. UW-Madison has generously agreed to support this project which provides anonymized MRIs of patients treated at the UW-Madison Carbone Cancer Center. The University of Wisconsin-Madison is a public land-grant research university in Madison, Wisconsin. The Wisconsin Idea is the university's pledge to the state, the nation, and the world that their endeavors will benefit all citizens.

In this competition, you’ll create a model to automatically segment the stomach and intestines on MRI scans. The MRI scans are from actual cancer patients who had 1-5 MRI scans on separate days during their radiation treatment. You'll base your algorithm on a dataset of these scans to come up with creative deep learning solutions that will help cancer patients get better care.

![image](https://lh5.googleusercontent.com/zbBUgbj1jyZxyu3r1vr5zKKr8yK1hSdwAM3HpD_n6j2W-5-wKP3ZRusi_3yskSgnC-tMRKqOEtLycbLkTWCJAUe4Cylv_VsW81DYI4ray02uZLeSnlzAuZRIU7L2Q0KURYSMqFI)

*In this figure, the tumor (thick pink line) is close to the stomach (thick red line). High doses of radiation are directed to the tumor while avoiding the stomach. The dose levels are represented by the rainbow of outlines, with higher doses represented by red and lower doses represented by green. *

Cancer takes enough of a toll. If successful, you'll enable radiation oncologists to safely deliver higher doses of radiation to tumors while avoiding the stomach and intestines. This will make cancer patients' daily treatments faster and allow them to get more effective treatment with fewer side effects and better long-term cancer control.

## 2. About this task
In this competition, we are segmenting organ cells in images. The training annotations are provided as RLE-encoded masks, and the images are in 16-bit grayscale PNG format.

Each case in this competition is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Some cases are split by time (early days are in train, later days are in test), while some cases are split by case - the entirety of the case is in train or test. The goal of this competition is to be able to generalize to both partially and wholly unseen cases.
So, the problem solved is multi-label segmentation tasks in the medical field.
## Tables of contents
[... Introduction](#introduction)  
[... Data analysis](#dataanalysis)  
[... Pre-processing step](#preprocessing)  
[..... RLE- run-length encode and decode]()
[..... CLAHE]()  
[..... Augmentation]()  
[... Segmentation model](#models)  
[... Measurement]()  
[..... IoU - Jaccard]()  
[..... Dice Coef]()  
[..... Tversky]()  
[..... Focal]()  
[... Optimizer]()  
[... Experiments]()  
## Data analysis<a name="dataanalysis"></a>
* The dataset has **115488** total images from **38496** total samples.
![data distributions](https://raw.githubusercontent.com/tuongkhangduongle/tuongkhangduongle.github.io/master/images/uwgi/__results___7_1.png)
From this figure, we can see the dataset is balanced. However, with masks segmented, the data is out of balance. The rate of the large bowel is highest, the rate of the small bowel is following to it, and the smallest is stomach.
![imablanced](https://raw.githubusercontent.com/tuongkhangduongle/tuongkhangduongle.github.io/master/images/uwgi/__results___12_1.png)
* It can be observed that more than half of the given examples have no annotations present!
    * There are 21,906 (56.9046%) examples with no annotations/masks/segmentation present
    * Inversely, there are 16,590 (43.0954%) examples with one or more annotations present
* There are 2,468 (6.41%) examples with one annotation present.
* It can be observed that the vast majority of single mask annotations are Stomach.
    * Of these annotations, 2286 (~92.6%) are Stomach
    * Of these annotations, 123 (~4.98%) are Large Bowel
    * Of these annotations, 59 (~2.39%) are Small Bowel
* There are 10,921 (28.37%) examples with two annotations present.
* It can be observed, in contrast to the single annotation examples, that the majority of annotations do NOT include stomach i.e. 'Large Bowel, Small Bowel'!
    * Of these annotations, 7781 (~71.3%) are 'Large Bowel, Small Bowel'
    * Of these annotations, 2980 (~27.3%) are 'Large Bowel, Stomach'.
    * Of these annotations, 160 (~1.47%) are 'Small Bowel, Stomach'.
* Finally, there are 3,201 (8.32%) examples with all three annotations present.

Here are some slices that I extracted from patient cases.
![visualized](https://raw.githubusercontent.com/tuongkhangduongle/tuongkhangduongle.github.io/master/images/uwgi/visualize.png)
## Pre-processing step <a name="preprocessing"></a>
### RLE - run length encode and decode
Run-length encoding is an algorithm for performing lossless data compression. Lossless data compression refers to compressing the data in such a way that the original form of the data can then be derived from it. When a character occurs a large number of times consecutively in a sequence, then we can represent the same consecutive subsequence using only a single occurrence of that character and its count. Using run-length encoding, we can save memory space while transmitting data and preserving its original form. It is useful when we want to store or transmit large sequences of data.

### K-Fold:
Cause of the distribution of the segmented slices meaning that not all patient slices are segmented, I use K-fold to avoid data leakage and stratify data with "empty" and "non-empty" masks.

### CLAHE
Contrast Limited Adaptive Histogram Equalization (CLAHE) is a variant of adaptive histogram equalization in which the contrast amplification is limited, so as to reduce this problem of noise amplification.

In CLAHE, the contrast amplification in the vicinity of a given pixel value is given by the slope of the transformation function. This is proportional to the slope of the neighbourhood cumulative distribution function (CDF) and therefore to the value of the histogram at that pixel value(from wikipedia).

Especially, CLAHE is used frequently in pre-processing medical images.
### Augmentation
Due to the unique properties of the field of medicine, such as we have not used Horizontal Flip to medical heart image because there is no such heart, in reality, leading to the overfitting problem, I used some augmentation techniques used commonly in medical images segmentation tasks.

## Segmentation Models <a name="models"></a>
### Unet Model
### Unet++ Model

## Model evaluate measurements
### IOU - Jaccard
### Tversky 
### Dice coef
### Focal 

## Optimizer
<img src="https://mlfromscratch.com/content/images/2019/12/saddle.gif" width=500>

## Experiments
