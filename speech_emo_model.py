import numpy as np
import _pickle as cPickle
from boruta import BorutaPy
from imblearn.combine import SMOTETomek
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier

np.random.seed(42)  # for reproducibililty

# Get data and label from saved file
f = open('D:/IIITD/iitkgp.pkl', 'rb')
data,label=cPickle.load(f)

# Min-max scaling
scal=MinMaxScaler()
data=scal.fit_transform(data,label)
x_train,x_test,y_train,y_test=train_test_split(data,label, test_size=1002, train_size=9013, random_state=42)
print(x_train.shape)

# Boruta feature selection
rf = RandomForestClassifier(random_state = 42)
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42,max_iter=90)
feat_selector.fit(x_train,y_train)

x_train=feat_selector.transform(x_train)
x_test=feat_selector.transform(x_test)

print(x_train.shape)
# Data resampling
smt = SMOTETomek(random_state=42)
x_train,y_train=smt.fit_sample(x_train,y_train)
print('1')
# Model
lr=SVC(kernel='rbf',C=100,gamma=0.1)
bag=BaggingClassifier(base_estimator=lr,n_estimators=20,bootstrap_features=True)
model=OneVsRestClassifier(bag)

model.fit(x_train, np.ravel(y_train))
preds=model.predict(x_test) 

score=cross_val_score(model,x_train,y_train,cv=KFold(10))
print('cv score',np.mean(score))
print('accuracy')
print(accuracy_score(np.ravel(y_test),preds))
print('report')
print(classification_report(np.ravel(y_test),preds))
print(confusion_matrix(np.ravel(y_test),preds))

    # convert label to binary class matrix 
   # Y_train = np_utils.to_categorical(y_train, nb_classes)
    #Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    #Y_test = np_utils.to_categorical(y_test, nb_classes)
    


    
