# Continuous Control Long MI EEG Investigation in Active BCI Systems
**Note**: The detailed explanations and protocols will be shared upon the acceptance and publication of the associated research paper.

## Overview
This project investigates the use of Motor Imagery (MI) Electroencephalography (EEG) signals in building an optimal MI-BCI (Brain-Computer Interface) system. Data was collected from 15 healthy participants performing Motor Imagery tasks (Hand, Feet, Tongue, Singing MI) for durations of 8-20 seconds. 

![EEG-Setup](https://github.com/M-Moeini/Continuous-Control-MI-EEG/blob/main/READMEImgs/EEG-Setup.jpg)

## 


**Recording Session**:
Hit the play to see!



[![EEG-Recording](path/to/thumbnail/image.png)](https://private-user-images.githubusercontent.com/79001628/398819334-02ca060d-524d-43c6-a69d-f96f13dfa2b7.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzUyNjA1NTcsIm5iZiI6MTczNTI2MDI1NywicGF0aCI6Ii83OTAwMTYyOC8zOTg4MTkzMzQtMDJjYTA2MGQtNTI0ZC00M2M2LWE2OWQtZjk2ZjEzZGZhMmI3Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjI3VDAwNDQxN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWVlZDQ3M2UyNjQyMzdjYjVlYTYyMTczMTMwOTkxZmQ4OGNhNTQwODcxMjk5NDUyNjI3MTIzZGJiNjY3MGQ5ZjcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.SnnC2B9MmHjAeqQVERFqv4DkYcv7z1On-kGemqh-iZA)


---

## Objectives

1. **Classification Accuracy**: Determine how accurately long MI-EEG signals can be classified.
2. **Paradigm Comparison**: Compare the performance of continuous versus switch control paradigms in MI-BCI systems.
3. **Participant Preferences**: Investigate which MI tasks participants prefer for long MI-BCI sessions.
4. **Optimal System Design**: Develop an online MI-BCI system that supports participant needs and achieves reliable classification accuracy.

---

## Experimental Protocol
![Experimental Protocol](https://github.com/M-Moeini/Continuous-Control-MI-EEG/blob/main/READMEImgs/EP.png)

---

## Results

1. **MI Tasks Comparison**:
   - Comparison of 4 MI tasks used in terms of classification accuracy.
   - Participant feedback on these MI tasks.

   ![Classification accuracy](https://github.com/M-Moeini/Continuous-Control-MI-EEG/blob/main/READMEImgs/MI-C.png)
   ![Participant feedback](https://github.com/M-Moeini/Continuous-Control-MI-EEG/blob/main/READMEImgs/TP-C.png)

2. **Training Data Requirements**:
   - Determine the minimum amount of training data needed for optimal performance.

   ![Training Data](https://github.com/M-Moeini/Continuous-Control-MI-EEG/blob/main/READMEImgs/TB-C.png)

3. **Epoching, classifiers and Number of Features Comparison**:
   -  Compare Epoching, classifiers and Number of Features.

   ![Training Data](https://github.com/M-Moeini/Continuous-Control-MI-EEG/blob/main/READMEImgs//WCF-C.png)


3. **Optimal Configuration**:
   - Best configuration for an online MI-BCI system based on classification accuracy, statistical analysis and the amount of time for training.

   ![Optimal Configuration](https://github.com/M-Moeini/Continuous-Control-MI-EEG/blob/main/READMEImgs/Optimal-BCI-C.png)

4. **Participant Results**:
   - Results summary for all participants on the optimal system.

   ![Participant Results](https://github.com/M-Moeini/Continuous-Control-MI-EEG/blob/main/READMEImgs/Optimal-BCI-R.png)

5. **Misclassification Chart**:
   - Visualization of misclassification trends.

   ![Misclassification Chart](https://github.com/M-Moeini/Continuous-Control-MI-EEG/blob/main/READMEImgs/MISS-R.png)

6. **Switch vs Continuous Comparison**:
   - Comparative results between the two paradigms.

   ![Switch vs Continuous](https://github.com/M-Moeini/Continuous-Control-MI-EEG/blob/main/READMEImgs/CS-C.png)
   ![Switch vs Continuous Table](https://github.com/M-Moeini/Continuous-Control-MI-EEG/blob/main/READMEImgs/CS-Table-C.png)

7. **Task Difficulty Ratings**:
   - Participant difficulty ratings results (scale: 1-5).

   ![TDRQ](https://github.com/M-Moeini/Continuous-Control-MI-EEG/blob/main/READMEImgs/TDRQ-C.png)


7. **Statistical Analysis Results**:
   Overall 5 different tests were used based on the purpose of the compariosn and the distribution of the data. ( Friedman, Nemenyi, Wilcoxon, Mann-Whitney U, Pearson)
   
   - **1- Compariosn of Differnt MI Tasks**:
      - **Friedman Test Results**:
    
   - **2- Compariosn of Taining Blocks Size**:
      - **Friedman Test Results**:

   - **3- Compariosn of Different Window Size**:
      - **Friedman Test Results**:
      - **Nemenyi Posthoc Test Results**:
    
   - **4- Compariosn of Different Classifiers**:
      - **Friedman Test Results**:
      - **Nemenyi Posthoc Test Results**:
        
   - **5- Compariosn of MI Tasks in the Optimal Configuration**:
      - **Friedman Test Results**:
   
   - **6- Compariosn of Feature Set Size**:
      - **Wilcoxon Test Results**:
   
   - **7- Compariosn of Task Difficulty Ratings**:
      - **Friedman Test Results**:
      - **Pearson Test Results**:
    
   - **8- Compariosn of Control Paradigm**:
      - **Mann-Whitney U Test Results**:




---

## Discussion
The discussion section will be made available after the paper submission.

---

## License and Copyright

### Copyright
This repository and its contents are copyright © Mahdi Moeini, 2024. All rights reserved.

### Usage Restrictions
Unauthorized use, reproduction, or distribution of this work is strictly prohibited. Any misuse, reproduction, or distribution without explicit written permission will be met with legal consequences to the fullest extent of the law.

---

**Note**: Additional details and protocols will be shared upon the acceptance and publication of the associated research paper.
