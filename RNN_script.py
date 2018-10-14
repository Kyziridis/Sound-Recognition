#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN_script: - Import Data 
         - Visualization 
         - MFCC_Feature extraction
         - Split train_set(70%) and test_set(30%)
         - Recurent Neural Network training
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
from tensorflow.python.ops import rnn, rnn_cell


path = sys.argv[1]
# Setting working directory
os.chdir(path)

# Import truth csv
my_data = np.matrix(np.genfromtxt('warblrb10k_public_metadata.csv', delimiter=',' , dtype=str , skip_header=1 ))



print("\nThis is a RNN script for bird sounds binary classification          >_")
print("_____________________________________________________________\n")


d = sys.argv[2]

        
# Set directory for wav files
os.chdir(d)

# Import wavs
l={}
r = []
name = []
dir_length=len(os.listdir(os.getcwd()))
print("\nImporting wav files...")
#pbar = tqdm(total=dir_length) # Specify the progressBar
for filename in tqdm(os.listdir(os.getcwd())):    
    x,sr = librosa.load(filename)    
    l[filename] = list()
    l[filename].append(x)
    r.append(x)
    name.append(filename.split('.')[0])

# Make raw as an np.array
raw = np.array(r)    

# Make truth as groundtruth with labels of wavs
ind = np.where(my_data[:,0] == name)[0] # find indexes of our wav in my_data
truth = np.matrix(my_data[ind]) # make truth only with these indexes
# Fix the order of truth
tmp = np.argsort(np.where(truth[:,0] == name)[1]) # index the mapping 
truth = truth[tmp] # set the actural order


# Splitting dataset into train and test | 70%/30% respectively
random.seed(1)
all_ind = np.where(truth[:,0])[0]
# Make train set contain 70% random observations of data
tr_ind = random.sample(list(all_ind) , round(70/100 * len(raw) ) )
#
tr_sound , tr_truth = raw[tr_ind] , truth[tr_ind]  
ts_sound , ts_truth = raw[np.delete(all_ind , tr_ind)] , truth[np.delete(all_ind , tr_ind)]


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
    


##########
# Extract mfcc feature by windowing for RNN
# Windowing
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

# Function for mel-frequency-spectogram-coefficinets with windowing for RNN
def extract_mfcc(sound, groundTruth, bands = 20, frames = 41):
    window_size = 512 * (frames - 1)
    mfccs = []
    labels = np.empty(0)
    for wav in tqdm(sound):        
        for (start,end) ,lab in zip( windows(wav , window_size) , groundTruth[:,1] ):
            if(len(wav[int(start):int(end)]) == window_size):
                signal = wav[int(start):int(end)]
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                mfccs.append(mfcc)
                labels = np.append(labels , lab)         
    features = np.asarray(mfccs).reshape(len(mfccs),frames,bands)
    return np.array(features) , np.array(labels,dtype = np.int)



""" Define the target hot_vector (bitmap) 2-d array col1 = 1 and col2 = 0 if NOT bird
 - Input: array of labels
 - Output: two column array with 1 where is true """
 
def hot_vector(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

    

# Make the input for the NN
print("\nConstructing the train_set...")
train_x , train_lab = extract_mfcc(tr_sound , tr_truth)
print("\nDone >_ \n")
print("Constructing the test_set...")
test_x , test_lab = extract_mfcc(ts_sound , ts_truth)
print("\nDone >_ \n")    

# Make hotvector of labels
tr_labels = hot_vector(train_lab)
ts_labels = hot_vector(test_lab)


# Recurent Neural Network
tf.reset_default_graph()

learning_rate = 0.01
training_iters = 1000
batch_size = 50
display_step = 200

# Network Parameters
n_input = 20 
n_steps = 41
n_hidden = 300
n_classes = 2 

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))


def RNN(x, weight, bias):
    stacked_rnn = []
    for iiLyr in range(2):
        stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple=True))
    MultiLyr_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
    output, state = tf.nn.dynamic_rnn(MultiLyr_cell, x, dtype = tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    return tf.nn.sigmoid(tf.matmul(last, weight) + bias)



prediction = RNN(x, weight, bias)

# Define loss and optimizer
loss_f = -tf.reduce_sum(y * tf.log(prediction))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss_f)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()



with tf.Session() as session:
    session.run(init)
    print("\nStart RNN_training, Iterations:"+str(training_iters)+", Learning Rate: "+str(learning_rate))
    for itr in tqdm( range(training_iters) ):    
        offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
        batch_x = train_x[offset:(offset + batch_size), :, :]
        batch_y = tr_labels[offset:(offset + batch_size), :]
        _, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y : batch_y})
            
        if itr % display_step == 0:
            # Calculate batch accuracy
            acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
            print ("\nIter " + str(itr) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    
    print('Test accuracy: ',round(session.run(accuracy, feed_dict={x: test_x, y: ts_labels}) , 3))
    print("\n    || Support GNU/Linux Foundation ||")









































