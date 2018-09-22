import numpy as np
import scipy.io.wavfile as wav
import os
import scipy.stats 
import _pickle as cPickle
from python_speech_features import mfcc,delta,ssc
numfeatper=7
mfc=13
dell=13
sscc=26
filtt=13
def featurex(filepath):
	# print(filepath)
	(rate,X) = wav.read(filepath)
	ceps = mfcc(X,rate)
	delt=delta(ceps,2)
	sscz=ssc(X,rate)
	filt=delta(delt,2)
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

	return source

def read_iitkgp(img_cols=(mfc+sscc+dell+filtt)*numfeatper):
	num = 10015
	solns=['A','D','F','H','N','S','SU']
	data=np.empty(shape=(num, img_cols))
	print(data.shape)
	label = []
	i=0
	rootdir = "F:/Sorted Database/anger/"
	for filename in os.listdir(rootdir):
		name = "".join(filename)
		full_name = rootdir+name
		data[i]=featurex(full_name)
		label.append(1)
		i=i+1
	rootdir = "F:/Sorted Database/disgust/"
	for filename in os.listdir(rootdir):
		name = "".join(filename)
		full_name = rootdir+name
		data[i]=featurex(full_name)
		label.append(2)
		i=i+1
	rootdir = "F:/Sorted Database/fear/"
	for filename in os.listdir(rootdir):
		name = "".join(filename)
		full_name = rootdir+name
		data[i]=featurex(full_name)
		label.append(3)
		i=i+1
	rootdir = "F:/Sorted Database/happy/"
	for filename in os.listdir(rootdir):
		name = "".join(filename)
		full_name = rootdir+name
		data[i]=featurex(full_name)
		label.append(4)
		i=i+1
	rootdir = "F:/Sorted Database/neutral/"
	for filename in os.listdir(rootdir):
		name = "".join(filename)
		full_name = rootdir+name
		data[i]=featurex(full_name)
		label.append(5)
		i=i+1
	rootdir = "F:/Sorted Database/sadness/"
	for filename in os.listdir(rootdir):
		name = "".join(filename)
		full_name = rootdir+name
		data[i]=featurex(full_name)
		label.append(6)
		i=i+1
	rootdir = "F:/Sorted Database/surprise/"
	for filename in os.listdir(rootdir):
		name = "".join(filename)
		full_name = rootdir+name
		data[i]=featurex(full_name)
		label.append(7)
		i=i+1
	label=np.array(label)
	f = open('D:/IIITD/iitkgp.pkl', 'wb')
	cPickle.dump((data, label), f)
	print(data.shape)
	print(label.shape)
	f.close()

read_iitkgp()

