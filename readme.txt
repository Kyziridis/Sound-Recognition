## README
Bird-sound classification neural network.

Folder Project contatins three scripts.py and one csv with the groundtruth values.

The two scripts named RNN_script.py and Simple_NN.py are suitable to be run in common laptops in python3.6 .

Script named RNN_Duranium.py is specified to import and train the neural network in super computer without visualization.

You can download the dataset from:
http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/



In order to run the two scripts you need:
1) Python3.6
2) sys
3) random
4) numpy 
5) os
6) librosa
7) matplotlib
8) tensorflow 
9) tqdm 

To run the scripts you have to specify first the the path of the project and then the path of the folder containing the wav files. Please don't forget the quots. Follow the instructions of the interactive script. 

For example if my Project is in my Desktop and the wavs are inside a wav folder under /Project/wav I run the script like this:

python RNN_script.py "/home/user/Desktop/Project" "/home/user/Desktop/Project/wav"