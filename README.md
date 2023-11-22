# UDT
This is the implementation of Unsupervised disease tags for automatic radiology report generation (https://doi.org/10.1016/j.bspc.2023.105742) at Biomedical Signal Processing and Control.

Datasets:

We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For IU X-Ray, you can download the dataset from https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing, and then put the files in data/iu_xray.

For MIMIC-CXR, you can download the dataset from https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing, and then put the files in data/mimic_cxr.

Run on IU X-Ray:

Run bash run_iu_xray.sh to train a model on the IU X-Ray data.

Run on MIMIC-CXR:

Run bash run_mimic_cxr.sh to train a model on the MIMIC-CXR data.
