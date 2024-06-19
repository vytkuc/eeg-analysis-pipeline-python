#Statistical and Machine learning algorithms for binary classification:
#0 - Binary Logistic Regression
#1 - Random Forest Classifier
#2 - XGBoost
#3 - Support Vector Machine
#4 - Artificial Neural Network    

from __future__ import division
import pandas as pd
import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import (accuracy_score, confusion_matrix, auc, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from statistics import stdev
from random import choice
from numpy import interp
import os

tf.get_logger().setLevel('ERROR')

#--------------------------------------
import mne
import os
#--------------------------------------

# Read the raw data
raw_no_med = mne.io.read_raw_fif("patient_2024_05_07_raw_no_med.fif", preload=True)
raw_with_med = mne.io.read_raw_fif("patient_2024_05_07_raw_with_med.fif", preload=True)

#Check information
display(raw_no_med.info)
display(raw_with_med.info)

#Plot the data
raw_no_med.plot();
raw_with_med.plot();

# Define the sampling frequency (same for both datasets)
fs = raw_no_med.info['sfreq']

#--------------------------------------

# Create fixed length epochs
epochs_no_med = mne.make_fixed_length_epochs(raw_no_med, duration=5.0, overlap=0.0, preload=True)
epochs_with_med = mne.make_fixed_length_epochs(raw_with_med, duration=5.0, overlap=0.0, preload=True)

print(epochs_no_med.get_data().shape)
print(epochs_with_med.get_data().shape)

#--------------------------------------

# Power Spectral Density (PSD)
psd_without_medication, freqs = mne.time_frequency.psd_array_welch(epochs_no_med.get_data(), sfreq=raw_no_med.info['sfreq'])
psd_with_medication, freqs = mne.time_frequency.psd_array_welch(epochs_with_med.get_data(), sfreq=raw_with_med.info['sfreq'])

print(psd_without_medication.shape)
print(psd_with_medication.shape)

#--------
# These results were for 2024-05-15 presentation

# Flatten the PSDs
#psd_without_medication = psd_without_medication.reshape(len(psd_without_medication), -1)
#psd_with_medication = psd_with_medication.reshape(len(psd_with_medication), -1)

#print(psd_without_medication.shape)
#print(psd_with_medication.shape)
      
# Combine the features and labels
#features = np.concatenate((psd_without_medication, psd_with_medication))
#labels = np.concatenate(([0] * len(psd_without_medication), [1] * len(psd_with_medication)))

#-------

# Calculate mean of PSD for each epoch and channel
mean_psd_without_medication = psd_without_medication.mean(axis=2)
mean_psd_with_medication = psd_with_medication.mean(axis=2)

print("Mean PSDs without medication shape:", mean_psd_without_medication.shape)
print("Mean PSDs with medication shape:", mean_psd_with_medication.shape)

# Combine the features and labels
features = np.concatenate((mean_psd_without_medication, mean_psd_with_medication))
labels = np.concatenate(([0] * len(mean_psd_without_medication), [1] * len(mean_psd_with_medication)))

print(features.shape)
print(labels.shape)

#--------------------------------------
'''
# Extract features
features_no_med = epochs_no_med.get_data().mean(axis=2)
features_with_med = epochs_with_med.get_data().mean(axis=2)

# Create labels
labels_no_med = [0] * len(features_no_med)
labels_with_med = [1] * len(features_with_med)

# Combine the features and labels
features = np.concatenate((features_no_med, features_with_med))
labels = np.concatenate((labels_no_med, labels_with_med))
'''
#--------------------------------------

# Split the data into training and test sets
#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, stratify=labels, shuffle=True)

X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33)

#--------------------------------------
#Prediction Accuracy
#--------------------------------------

base_fpr = np.linspace(0, 1, 101)

def predictionR(classifier, X_train, X_test, y_train, y_test):
    pipe = make_pipeline(StandardScaler(), classifier)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return y_test, y_pred, pipe

def evaluationR(y, y_hat, title = 'Confusion Matrix'):
    cm = confusion_matrix(y, y_hat, labels=[0.0, 1.0])
    sensitivity = cm[0,0]/(cm[0,0] + cm[0,1])
    specificity = cm[1,1]/(cm[1,1] + cm[1,0])
    accuracy = accuracy_score(y, y_hat)
    fpr, tpr, thresholds = roc_curve(y, y_hat, pos_label=1)
    AUC = auc(fpr, tpr)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    return accuracy, sensitivity, specificity, AUC, tpr

def print_accuracy(res_acc, res_sens, res_spec, res_AUC, tprs):
    print("%4.2f  ±%4.2f    %4.2f ±%4.2f   %4.2f ±%4.2f   %4.2f ±%4.2f" % (100*sum(res_acc)/len(res_acc), 100*stdev(res_acc), 100*sum(res_sens)/len(res_sens), 100*stdev(res_sens),
          100*sum(res_spec)/len(res_spec), 100*stdev(res_spec), sum(res_AUC)/len(res_AUC), stdev(res_AUC)))
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    plt.figure()
    plt.plot(base_fpr, mean_tprs, 'b')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.text(x = 0.5, y = 0.2, s="AUC = %4.4f" % (sum(res_AUC)/len(res_AUC)))
    #plt.show()

scaler = StandardScaler()

#--------------------------------------
#Artificial Neural Network
#--------------------------------------

def ANN(X_train, X_test, y_train, y_test, scaler):
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    lb = LabelBinarizer().fit(y_train)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)

    model = Sequential()
    model.add(Dense(30, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=0)
    predictions = model.predict(X_test)
    predictions = lb.inverse_transform(predictions)
    y_test = lb.inverse_transform(y_test)
    return y_test, predictions

#----------------------------------------------------------------------------------------
#Methods for classification
#----------------------------------------------------------------------------------------

def binary_classification(X, y, N = 10, scaler = scaler):
  
    res_acc =  [[], [], [], [], [], [], [], []]
    res_sens = [[], [], [], [], [], [], [], []]
    res_spec = [[], [], [], [], [], [], [], []]
    res_AUC =  [[], [], [], [], [], [], [], []]
    tprs =     [[], [], [], [], [], [], [], []]
    
    
    for i in range(N):
        print("Run %d" %(i))

        #split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        #scaling  
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        #----------------------------------------------
        #Classification Methods
        #----------------------------------------------
        #Binary Logistic Regression
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        #print(y_test)
        #print(y_pred)
        
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[0].append(acc)
        res_sens[0].append(sens)
        res_spec[0].append(spec)
        res_AUC[0].append(AUC)
        tprs[0].append(tpr)

        #Random Forest Classifier
        y_test, y_pred, model = predictionR(RandomForestClassifier(), X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[1].append(acc)
        res_sens[1].append(sens)
        res_spec[1].append(spec)
        res_AUC[1].append(AUC)
        tprs[1].append(tpr)
        
        #XGBoost
        y_test, y_pred, model = predictionR(XGBClassifier(), X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[2].append(acc)
        res_sens[2].append(sens)
        res_spec[2].append(spec)
        res_AUC[2].append(AUC)
        tprs[2].append(tpr)
          
        #Support Vector Machine
        y_test, y_pred, model = predictionR(SVC(), X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[3].append(acc)
        res_sens[3].append(sens)
        res_spec[3].append(spec)
        res_AUC[3].append(AUC)
        tprs[3].append(tpr)

        #Artificial Neural Network
        y_test, y_pred = ANN(X_train, X_test, y_train, y_test, scaler)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[4].append(acc)
        res_sens[4].append(sens)
        res_spec[4].append(spec)
        res_AUC[4].append(AUC)
        tprs[4].append(tpr)
        
    print("Accuracy %  Sensitivity %  Specificity %  AUC")

    #LogReg
    print("\nBinary Logistic Regression")
    print_accuracy(res_acc[0], res_sens[0], res_spec[0], res_AUC[0], tprs[0])
    #RandomForest
    print("\nRandom Forest Classifier")
    print_accuracy(res_acc[1], res_sens[1], res_spec[1], res_AUC[1], tprs[1])
    #XGBoost
    print("\nXGBoost")
    print_accuracy(res_acc[2], res_sens[2], res_spec[2], res_AUC[2], tprs[2])
    #Support Vector Machine
    print("\nSupport Vector Machine")
    print_accuracy(res_acc[3], res_sens[3], res_spec[3], res_AUC[3], tprs[3])
    #Artificial Neural Network
    print("\nArtificial Neural Network")
    print_accuracy(res_acc[4], res_sens[4], res_spec[4], res_AUC[4], tprs[4])
    
    '''
    with open(os.path.join(datadir, "ACC.npy"), 'wb') as f:
        np.save(f, res_acc)
    with open(os.path.join(datadir, "SENS.npy"), 'wb') as f:
        np.save(f, res_sens)
    with open(os.path.join(datadir, "SPEC.npy"), 'wb') as f:
        np.save(f, res_spec)
    with open(os.path.join(datadir, "AUC.npy"), 'wb') as f:
        np.save(f, res_AUC)
    with open(os.path.join(datadir, "TPRS.npy"), 'wb') as f:
        np.save(f, tprs)
    '''


N_runs=50


binary_classification(X, y, N_runs)


