# Video-based detection of task-related concentration using advanced machine learning

This repository contains the codebase that was used to conduct our experiments to measure task related concentration with the combined use of a conventional digital video camera and machine learning. Our experiments results proved that machine learning models with the use of the contactless video photoplethysmography (VPPG) methodology produced high accuracy (around 97%), and outperformed the invasive electrocardiogram (ECG) based method to a great degree. 

Our experiments compare the performance of three machine learning methods: Support Vector Machine (SVM), XGBoost, and pre-trained VGG on both VPPG signals and ECG signals. Pre-trained VGG based on the VPPG signals achieved the best results. 

We further test the performance of an ensemble method based on three machine learning methods. Even though the ensembled models performed better than did each individual method, a VGG based model alone was sufficiently accurate for detecting task related concentration.


## Data processing
run `extract_facial_frames.py` to extract frames from videos.

run `facial_data_process.py` to create csv files and datasets. Rest samples are labeled as "0" and Task samples are labeled as "1".

run `ecg_data_process.py` to create ECG data.


## On-Task Detection
##### Method 1: Stress detection using facial images	
run `main_facial_images.py`

##### Method 2: Stress detection using ECG signals.  
run `main_ECG.py`

##### Method 3: Weighted average ensemble.
run `main.py` to produce all results.
	
