import numpy as np
import os
import os.path
import librosa
import _pickle as cPickle
import scipy
from python_speech_features import mfcc,delta,ssc             
numfeatper=7
mfc=13
dell=13
sscc=26
filtt=13
#dssc=26


def featurex(filepath):
    (X,rate) = librosa.load(filepath,sr=48000)
    ceps = mfcc(X,rate,nfft=2048)
    delt=delta(ceps,2)
    sscz=ssc(X,rate,nfft=2048)
    filt=delta(delt,2)
    #zeroo=delta(sscz,2)
    #librosa.feature.zero_crossing_rate(X,rate)
#    zeroo=zeroo.reshape((zeroo.shape[1],zeroo.shape[0]))
    ls = []
    for i in range(ceps.shape[1]):
        temp = ceps[:,i]
        lfeatures  = [np.mean(temp), np.var(temp), np.amax(temp), np.amin(temp),scipy.stats.kurtosis(temp), scipy.stats.skew(temp),scipy.stats.iqr(temp)]
        temp2 = np.array(lfeatures)
        ls.append(temp2)

        
    ls2=[]
    for i in range(delt.shape[1]):
        dtemp = delt[:,i]
        dlfeatures  = [np.mean(dtemp), np.var(dtemp), np.amax(dtemp), np.amin(dtemp),scipy.stats.kurtosis(dtemp), scipy.stats.skew(dtemp),scipy.stats.iqr(dtemp)]
        dtemp2 = np.array(dlfeatures)
        ls2.append(dtemp2)
    ls3=[]
    for i in range(sscz.shape[1]):
        stemp = sscz[:,i]
        slfeatures  = [np.mean(stemp), np.var(stemp), np.amax(stemp), np.amin(stemp),scipy.stats.kurtosis(stemp), scipy.stats.skew(stemp),scipy.stats.iqr(stemp)]
        stemp3 = np.array(slfeatures)
        ls3.append(stemp3)
    
    ls4=[]
    for i in range(filt.shape[1]):
        ftemp=filt[:,i]
        flfeatures=[np.mean(ftemp), np.var(ftemp), np.amax(ftemp), np.amin(ftemp),scipy.stats.kurtosis(ftemp), scipy.stats.skew(ftemp),scipy.stats.iqr(ftemp)]
        ftemp4 = np.array(flfeatures)
        ls4.append(ftemp4)
    
    
    source = np.array(ls).flatten()
    source = np.append(source, np.array(ls2).flatten())
    source = np.append(source, np.array(ls3).flatten())
    source = np.append(source, np.array(ls4).flatten())
#    source = np.append(source, np.array(ls5).flatten())
    return source


def read_emodb(img_cols=(mfc+sscc+dell+filtt)*numfeatper):
    rootdir = "C:/Users/ANJALI/Downloads/Audio_Speech_Actors_01-24/"
    num = 1440
    solns=['1','2','3','4','5','6','7','8']
    data = np.empty(shape=(num, img_cols))
    print(data.shape)
    label = np.empty(num, dtype=int)
    i = 0
    for filename in os.listdir(rootdir):
        name = "".join(filename)
        full_name = rootdir+name
        data[i] = featurex(full_name)
        label[i] = solns.index(filename[7])
        i = i+1
    f = open('C:/Users/ANJALI/Anaconda3/envs/anjenv/Lib/site-packages/emodbdata/ravdess.pkl', 'wb')
    cPickle.dump((data, label), f)
    f.close()
    print(data.shape)

read_emodb()





