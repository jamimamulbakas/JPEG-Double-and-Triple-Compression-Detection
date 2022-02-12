# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:42:44 2019

@author: NIT
"""
from __future__ import print_function
import tensorflow as tf
import pickle
import os
import numpy as np
import scipy.io as sio
import h5py
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score
from keras.utils import to_categorical
from keras.models import model_from_json
import scipy.io as sio

#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
save_path = './90_50_90/checkpoints/'
model_name = 'my_model_1000.h5'
top_model_name='best_model.h5'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
#==================================================
# Test Data
ftest = h5py.File('./90_50_90/multiple_forgery/dresden/dresden_multi_forged.mat')
Att = {}
for ktt, vtt in ftest.items():
    Att[ktt] = np.array(vtt)
Btt = list(Att.values())
arrtt = np.asarray(Btt)
arrt1t = arrtt.reshape((231,90768))  #90768 # 2337
x_test = np.transpose(arrt1t)
#label data

test_dataset=x_test#, test_labels= shuffle(x_test, trYst)
#=================================================

learning_rate = 0.0001
training_iters = 2804400
batch_size = 200
display_step = 10
epoch=2

# Network Parameters
n_input = 231 
n_classes = 3  # single,double,triple
dropout = 0.40  


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
test_dataset = np.reshape(test_dataset, [-1, 231, 1, 1])


model = models.Sequential()
model.add(layers.Conv2D(100, (3, 1), activation='relu', input_shape=(231, 1, 1)))
model.add(layers.MaxPooling2D((2, 1)))
model.add(layers.Conv2D(100, (3, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 1)))
model.add(layers.Conv2D(100, (3, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 1)))
model.add(layers.Conv2D(100, (3, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 1)))


model.add(layers.Flatten())
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(3, activation='softmax'))
model.summary()



save_path_full = os.path.join(save_path, model_name)
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(save_path_full)
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(test_dataset, test_labels, verbose=0)
pred = loaded_model.predict(test_dataset)
prediction = np.argmax(pred,1)
prob=prediction.reshape(244,372) # (41,57)
sio.savemat('./90_50_90/multiple_forgery/dresden/dresden_multi_pM_90_50_90.mat',{'probability':prob})
#print("Test %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))






