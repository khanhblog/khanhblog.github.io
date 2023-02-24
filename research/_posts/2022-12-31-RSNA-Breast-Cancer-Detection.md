---
layout: post
title: RSNA Breast Cancer Detection - Overview and methods.
comment: True
---

This is a competition that I joined in Kaggle. The field of it is computer vision, especially medical images (the topic that I pay a lot of attention to). The following is my research, insight and experiment in the duration of solving problem in this competition. 

---------
## Table of contents
[... 1. Descriptions](#descriptions)  
[... 2. First sight of the dataset](#firstsight)  

---------


## 1. Descriptions<a name="descriptions"></a>
### Overview
According to the WHO, breast cancer is the most commonly occurring cancer worldwide. In 2020 alone, there were 2.3 million new breast cancer diagnoses and 685,000 deaths. Yet breast cancer mortality in high-income countries has dropped by 40% since the 1980s when health authorities implemented regular mammography screening in age groups considered at risk. Early detection and treatment are critical to reducing cancer fatalities, and your machine learning skills could help streamline the process radiologists use to evaluate screening mammograms.

Currently, early detection of breast cancer requires the expertise of highly-trained human observers, making screening mammography programs expensive to conduct. A looming shortage of radiologists in several countries will likely worsen this problem. Mammography screening also leads to a high incidence of false positive results. This can result in unnecessary anxiety, inconvenient follow-up care, extra imaging tests, and sometimes a need for tissue sampling (often a needle biopsy).

The competition host, the Radiological Society of North America (RSNA) is a non-profit organization that represents 31 radiologic subspecialties from 145 countries around the world. RSNA promotes excellence in patient care and health care delivery through education, research, and technological innovation.

Your efforts in this competition could help extend the benefits of early detection to a broader population. Greater access could further reduce breast cancer mortality worldwide. 

### Goal of the competition
The goal of this competition is to identify breast cancer. You'll train your model with screening mammograms obtained from regular screening.

Your work improving the automation of detection in screening mammography may enable radiologists to be more accurate and efficient, improving the quality and safety of patient care. It could also help reduce costs and unnecessary medical procedures.

*(via [Kaggle: RSNA Screening Mammography Breast Cancer Detection 2022](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview))*

## 2. First sight of the dataset<a name='firstsight'></a>
### Metadata and images
The dataset includes metadata and images (about 54000 images labeled positive and negative). In one hand, the ones in training data concludes atributes as:

* site_id	
* patient_id
* image_id	
* laterality
* view
* age	
* cancer
* biopsy
* invasive
* BIRADS
* implant
* density
* machine_id
* difficult_negative_case     

In the other hand, the metadata in test set has:

* site_id
* patient_id
* image_id
* laterality
* view
* age
* implant
* machine_id
* prediction_id

After that, I have some insights as:
* In the metadata training file we have plenty of missing values.
* Repeated values for patient_id. It seems that for each patient, 4 images have been taken.
* Some features from the training set do not appear in the testing one.

In terms of images, the format is dicom (I change them to jpeg format for processing)

``` Python
def process(f, size=256, save_folder="", extension="png"):
    patient = f.split('/')[-2]
    image = f.split('/')[-1][:-4]

    img = pydicom.dcmread(f).pixel_array
    img = (img - img.min()) / (img.max() - img.min())
    
    img = cv2.resize(img, (size, size))

    cv2.imwrite(save_folder + f"{patient}_{image}.{extension}", (img * 255).astype(np.uint8))
_ = Parallel(n_jobs=4)(

_ = Parallel(n_jobs=4)(
    delayed(process)(uid, size=SIZE, save_folder=SAVE_FOLDER, extension=EXTENSION)
    for uid in tqdm(train_images)
)
```

### Site and patient

Starting with the hospital, we can observe that we only have two of them in our dataset. Apart from that, we can observe that background colors depend on the site where the image was taken.

``` Python 
dcm_path = "./train_images/" ### images folder

def images_site(site_id):
    ids = train[train.site_id == site_id]['patient_id'].unique()
    for i, id_ in enumerate(ids[[0,3]]):
        patient_path = dcm_path + str(id_) +'/'
        fig = plt.figure(figsize = (22,5))
        for j, file in enumerate(listdir(patient_path)):
            plt.subplot(1, 4, j+1)
            dataset = pydicom.dcmread(patient_path + file)
            p = plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
            plt.axis('off');

print('There are {} different hospitals in the dataset.\n'.format(len(train.site_id.unique())))            
for val in train.site_id.unique(): 
    images_site(val)
```

![images1](https://raw.githubusercontent.com/tuong-khang/tuong-khang.github.io/master/images/rsna2022/site_pat_01.png)
![images2](https://raw.githubusercontent.com/tuong-khang/tuong-khang.github.io/master/images/rsna2022/site_pat_02.png)

Also, the number of entries per patient has different as:

![images3](https://raw.githubusercontent.com/tuong-khang/tuong-khang.github.io/master/images/rsna2022/each_img_pat.png)

From the table, 4 images per patient is the most common frequency. However, we saw that there is a big number of them having more than 4 images. Furthermore, a patient has rarely more images asociated.

### View and age of patients
* We almost have the same amount of left breast images than right ones.
* Ver few values under 40 years old for Age. Some peaks between 50 and 70 year-old.
* Six different values for view feature. Quite imbalanced (CC and MLO are the most common ones).

Laterality feature indicates whether the image is of the left or right breast. This issue can be fixed when processing images. View instead, refers to the orientation of the image. The default for a screening exam is to capture two views per breast. That's the reason for having almost the same amount of left and right breast images.

![image4](https://raw.githubusercontent.com/tuong-khang/tuong-khang.github.io/master/images/rsna2022/view_age_pat.png)

## Cancer, Biopsy, Invasive and BIRADS

