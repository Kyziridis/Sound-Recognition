#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NN_script: - Importing data
           - Visualization
           - Splitting data into train and test
           - Feature extraction(train/test): mfccs, chroma, mel, contrast, tonnetz
           - Four-layer Neural Network train for binnary classification
"""
import sys
import random
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

path = "/home/dead/Documents/API/Birds/Project"
# Setting working directory
os.chdir(path)

# Import truth csv
my_data = np.matrix(np.genfromtxt('warblrb10k_public_metadata.csv', delimiter=',' , dtype=str , skip_header=1 ))



print("\nThis is a simple four-layer NN script for bird sounds binary classification          >_")
print("______________________________________________________________________________\n")

# Second statement
d = "/home/dead/Documents/API/Birds/train150"        
# Set directory for wav files
os.chdir(d)

# Import wavs
l={}
r = []
name = []
i = 1
dir_length=len(os.listdir(os.getcwd()))
print("\nImporting wav files...")
#pbar = tqdm(total=dir_length) # Specify the progressBar
for filename in tqdm(os.listdir(os.getcwd())):    
    #pbar.update()
    x,sr = librosa.load(filename)    
    l[filename] = list()
    l[filename].append(x)
    r.append(x)
    name.append(filename.split('.')[0])
    #print(" - Import file: " , i , str(" out of _") , str(dir_length) ,str("SampleRate: ") ,str(sr))
    i = i+1
# Make raw as an np.array
raw = np.array(r)    


# Make truth as groundtruth with labels of wavs
ind = np.where(my_data[:,0] == name)[0] # find indexes of our wav in my_data
truth = np.matrix(my_data[ind]) # make truth only with these indexes
# Fix the order of truth
tmp = np.argsort(np.where(truth[:,0] == name)[1]) # index the mapping 
truth = truth[tmp] # set the actural order


# Function for plotting waves    
def ploting_wave(sound):    
    length=len(sound)
    i=1
    plt.figure()
    for freq in sound:        
        plt.subplot(length,1,i)
        librosa.display.waveplot(freq , sr=22050)    
        i += 1
    plt.show()
    
    
# Function for spectograms        
def plot_specgram(sound):
    i = 1
    length =len(sound)
    plt.figure()
    for f in sound:
        plt.subplot(length,1,i)
        specgram(f, Fs=22050)
        i += 1
    plt.show()    


# Function for logspectograms
def plot_logspecgram(sound):
    i = 1
    length = len(sound)
    plt.figure()
    for f in sound:
        plt.subplot(length,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        i += 1    
    plt.show() 


# Feature extraction with librosa
def extract_feature(sound):
    stft = np.abs(librosa.stft(sound))
    mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(sound, sr=sr).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sr).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


# Concatenate all names , features and labels for each wav file
def concatenate(sound , groundTruth):
    features, labels , names = np.empty((0,193)), np.empty(0) , np.empty(0)
    print("\nConcatenating: names, features and labels for each wav file....")
    pbar = tqdm(total=len(sound)) # Specify the progressBar    
    for fn,name,lab in zip(sound,groundTruth[:,0],groundTruth[:,1]):        
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        labels = np.append(labels, lab)
        names = np.append(names , name)
        pbar.update()
    pbar.close()    
    return np.array(names , dtype =str), np.array(features), np.array(labels, dtype = np.int)

names, features , label = concatenate(raw , truth)
print("\nVariables: names , features and label \n")
print("Features: melspectrogram \n  \t mfcc \n \t chorma-stft \n \t spectral_contrast \n \t tonnetz \n")




""" Define the target hot_vector (bitmap)
2-d array col1 = 1 and col2 = 0 if NOT bird

 - Input: array of labels
 - Output: two column array with 1 where is true """
def hot_vector(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


#PLOTTING
# Plot the five first waves
print("\n")
flag1 = input("Visulize some of the WAVE_Plots? [y/n]        >_ ")
if flag1 == 'Y' or flag1 == 'y' or flag1 == 'yes':    
    f1 = input("\nSpecify how many plots to visualize: (number)  >_")
    print("\nWave_Plots")
    ploting_wave(raw[0:int(f1)])    
    print("\n")

# Plot the five first spectograms
flag2 = input("Visualize some of the spectograms? [y/n]      >_ ")
if flag2 == 'Y' or flag2 == 'y' or flag2 == 'yes':
    f2 = input("\nSpecify how many plots to visualize: (number)  >_")
    print("Spectograms")
    plot_specgram(raw[0:int(f2)])    
    print("\n")

flag3 = input("Visualize some of the LogSpectograms? [y/n]   >_  ")
if flag3 == 'Y' or flag3 == 'y' or flag3 == 'yes':
    f3=input("\nSpecify how many plots to visualize: (number)  >_")
# Plot the first five logspectograms
    print("\nLogSpectograms")
    plot_logspecgram(raw[0:int(f3)])




# Splitting dataset into train and test | 70%/30% respectively
random.seed(1)
all_ind = np.where(truth[:,0])[0]

# Make train set contain 70% random observations of data
tr_ind = random.sample(list(all_ind) , round(70/100 * len(names)))
train_nam , train_x , train_lab = names[tr_ind] , features[tr_ind] , label[tr_ind]    

# Make test set contain the rest 30%
test_nam , test_x , test_lab = names[np.delete(all_ind , tr_ind)], features[np.delete(all_ind , tr_ind)] ,label[np.delete(all_ind , tr_ind)]

# Make hotvector of labels
tr_labels = hot_vector(train_lab)
ts_labels = hot_vector(test_lab)


# NN parameters
training_epochs = 1500
n_dim = train_x.shape[1]
n_classes = 2
n_hidden_units_one =400
n_hidden_units_two = 430
n_hidden_units_three = 470
n_hidden_units_four = 500
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01




print("\n Importing, Plotting and Feature extraction procedure accomplished.\n")
print("\n")
nnflag = input("Start Neural Network Training? [Y/n] :  >_ ")
if nnflag == 'Y' or nnflag == 'y' or nnflag == 'yes':

    print("\n NN with:  " , str(training_epochs) , ("iterations and  ") , str(learning_rate) , ("learning rate\n"))
    print("\n")
    
    """Define weights and biases for hidden and output layers of the network. 
       Use the sigmoid function in the first hidden layer and tanh in the second hidden layer.
       The output layer has softmax . """
    # Neural NET
    
    # Placeholder for input and output   
    X = tf.placeholder(tf.float32,[None,n_dim])
    Y = tf.placeholder(tf.float32,[None,n_classes])
    
    # First Layer
    W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
    h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)
    
    # Second Layer
    W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
    h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)
        
    # Third layer
    W_3 = tf.Variable(tf.random_normal([n_hidden_units_two,n_hidden_units_three], mean = 0, stddev=sd))
    b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean = 0, stddev=sd))
    h_3 = tf.nn.sigmoid(tf.matmul(h_2,W_3) + b_3)
    
    # Fourth layer
    W_4 = tf.Variable(tf.random_normal([n_hidden_units_three,n_hidden_units_four], mean = 0, stddev=sd))
    b_4 = tf.Variable(tf.random_normal([n_hidden_units_four], mean = 0, stddev=sd))
    h_4 = tf.nn.sigmoid(tf.matmul(h_3,W_4) + b_4)
    
    # Output Layer
    W = tf.Variable(tf.random_normal([n_hidden_units_four,n_classes], mean = 0, stddev=sd))
    b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
    y_ = tf.nn.sigmoid(tf.matmul(h_4,W) + b)
    
    init = tf.global_variables_initializer()
    
       
    """ - Cross-entropy cost function, using gradient descent optimizer. 
        - Initialize cost function and optimizer.
        - Define and initialize variables for accuracy calculation of the prediction by model."""
    
    cost_function = -tf.reduce_sum(Y * tf.log(y_))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
    
    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
        
    """Train the NN"""
    
    cost_history = np.empty(shape=[1],dtype=float)
    y_true, y_pred = None, None
    with tf.Session() as sess:
        sess.run(init)
        print("\nTraining the NN")
        for epoch in tqdm( range(training_epochs) ):            
            _,cost = sess.run([optimizer,cost_function],feed_dict={X:train_x,Y:tr_labels})
            cost_history = np.append(cost_history,cost)
        
        y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: test_x})
        y_true = sess.run(tf.argmax(ts_labels,1))
        print("\n")
        print('Test-accuracy:',round(sess.run(accuracy, feed_dict={X: test_x, Y: ts_labels}) , 3))
    
    
    
    p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print ("F-Score:      ", round(f,3))
    print("\n  Support GNU/Linux ||-Free Software Foundation-|| >_ \n")
    
    plt.figure()
    plt.plot(cost_history)
    #v = [0,training_epochs,0,np.max(cost_history)]
    #plt.axis(v)
    plt.show()











































