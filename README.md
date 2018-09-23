# Bagging-With-Strong-Learners-for-Emotion-Recognition-from-Speech

Speech emotion recognition, a highly promising and exciting problem in the field of Human Computer Interaction, has been studied and analyzed over several years, and concerns the task of recognizing a speaker's emotions from their speech recordings. Recognizing emotions from speech can go a long way in determining a person's physical and psychological state of well-being, and also provides us with valuable information regarding language etc. In this work we analyzed three corpora - the Berlin EmoDB, the Indian Institute of Technology Kharagpur Simulated Emotion Hindi Speech Corpus (IITKGP-SEHSC) and the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), and extracted spectral features from them which were further processed and reduced to the required feature set by using Boruta, a wrapper-based feature selection method. We propose a bagged ensemble comprising of Support Vector Machines with a Gaussian kernel as a viable algorithm for the problem at hand, and report the results obtained on the three datasets above.

# Results

![alt text](https://github.com/cpankajr/Bagging-With-Strong-Learners-for-Emotion-Recognition-from-Speech/blob/master/results.png)

#
**Note: All above codes designed for IITKGP database , same codes can be used for other databses with slight modification _speech_emotion_model.py_ in code. I cannot share IITKGP-SEHSC database because it is a property of IITKGP. to get the database please contact [Prof. K. S. Rao](http://cse.iitkgp.ac.in/~ksrao/).**

**Download [_EmoDB_](https://drive.google.com/open?id=16LtD3SQuMral216YyB6pyRd6DXHWhWTX) database, [_Ravdess_](https://drive.google.com/drive/folders/118PZOuTN_2a-PwyQS1ZDZ7ZjnNoSoOS9?usp=sharing) database**

# Libraries required 

 - Scipy
 - Numpy
 - python_speech_features 
 - boruta
 - imblearn
 - sklearn

#
MFCCs nd their delta nd double delta features are calculated for each frame. (13+13+13=39)

The spectral sub-band centroids are calculated next, 26 for each frame.

the mean, variance, maximum value, minimum value, skewness, kurtosis and inter-quartile range. These values were calculated for each audio file over all the frames and for each coefficient, which gave us a feature vector of dimension \((13+13+13+26) * 7 = 455 features for each sample

**read_iitkgp.py** -> for extracting features for IITKGP dataset. You can download output pkl file [**_iitkgp.pkl_**](https://drive.google.com/file/d/13zgi98yDhnyfItsExnnRy3m8xRWzncQ8/view?usp=sharing)

**readdataemodb.py** ->	for extracting features for EmoDB dataset. You can download output pkl file [**_emodb.pkl_**](https://drive.google.com/file/d/11lHqfcD4r99RfR275r2RZiJnDfL1fngt/view?usp=sharing)

**readdataravdess.py** -> for extracting features for Ravdess dataset.

Our base estimator was a Support Vector Machine with a Gaussian kernel, penalty term 100 and kernel coefficient 0.1. We combined 20 of these in a bagging ensemble, and set bootstrap features as True - so samples are drawn from the training set with replacement. This entire procedure was carried out using the scikit-learn Python package.

**speech_emotion_model.py** -> for training and testing a model
