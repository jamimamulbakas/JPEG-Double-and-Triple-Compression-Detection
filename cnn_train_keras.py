
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
from sklearn.metrics import confusion_matrix, classification_report, auc, precision_recall_curve
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import xlsxwriter
from keras.models import model_from_json




save_path = './checkpoints_pr_rc/'  #'./50_70_90/checkpoints/'
model_name = 'my_model_paper_50_70_90.h5'
top_model_name='best_model_with_drop1.h5'
if not os.path.exists(save_path):
    os.makedirs(save_path)
#pretrain=False
#train data
ft = h5py.File('./50_70_90/train_data_90_50.mat')
At = {}
for kt, vt in ft.items():
    At[kt] = np.array(vt)
Bt = list(At.values())
arrt = np.asarray(Bt)
arrt1 = arrt.reshape((231,2804400))   # 2804400 for triple # 1869600 For double
x_train = np.transpose(arrt1)
#label data
f = h5py.File('./50_70_90/train_label_90_50.mat')
A = {}
for k, v in f.items():
    A[k] = np.array(v)

B = list(A.values())
arr = np.asarray(B)
arr1 = arr.reshape((1,2804400))
ttY1 = np.transpose(arr1)

trY=np.array(ttY1).astype(np.int32)
train_dataset, train_labels= shuffle(x_train, trY)
#train_labels = to_categorical(train_labels).astype(np.float32)   # Use when binary_crossentropy loss is used
#==================================================
# Test Data
ftest = h5py.File('./50_70_90/test_data_90_50.mat')
Att = {}
for ktt, vtt in ftest.items():
    Att[ktt] = np.array(vtt)
Btt = list(Att.values())
arrtt = np.asarray(Btt)
arrt1t = arrtt.reshape((231,701100))  #701100 for triple #  467400 For double 
x_test = np.transpose(arrt1t)
#label data
fst = h5py.File('./50_70_90/test_label_90_50.mat')
Ast = {}
for kst, vst in fst.items():
    Ast[kst] = np.array(vst)

Bst = list(Ast.values())
arrst = np.asarray(Bst)
arr1st = arrst.reshape((1,701100))   # 701100
ttY1st = np.transpose(arr1st)

trYst=np.array(ttY1st).astype(np.int32)
test_dataset, test_labels= shuffle(x_test, trYst)
#test_labels = to_categorical(test_labels).astype(np.float32)   # Use when binary_crossentropy loss is used
#=================================================

learning_rate = 0.0001
training_iters = 2804400
batch_size = 400
display_step = 10
epoch=5

# Network Parameters
n_input = 231 
n_classes = 3  # single,double,triple
dropout = 0.40  


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
train_dataset = np.reshape(train_dataset, [-1, 231, 1, 1])
test_dataset = np.reshape(test_dataset, [-1, 231, 1, 1])





save_path_full = os.path.join(save_path, model_name)


# Code according to sumana paper
model = models.Sequential()
model.add(layers.Conv2D(100, (3, 1), strides=(1, 1), activation='relu', input_shape=(n_input, 1, 1)))
model.add(layers.MaxPooling2D((2, 1), strides=(2, 1)))
model.add(layers.Conv2D(100, (3, 1), strides=(1, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 1),strides=(2, 1)))

model.add(layers.Conv2D(100, (3, 1), strides=(1, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 1),strides=(2, 1)))
model.add(layers.Conv2D(100, (3, 1), strides=(1, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 1),strides=(2, 1)))

#My code
#model = models.Sequential()
#model.add(layers.Conv2D(100, (3, 1), activation='relu', input_shape=(231, 1, 1)))
#model.add(layers.MaxPooling2D((2, 1)))
#model.add(layers.Conv2D(100, (3, 1), activation='relu'))
#model.add(layers.MaxPooling2D((2, 1)))
#model.add(layers.Conv2D(100, (3, 1), activation='relu'))
#model.add(layers.MaxPooling2D((2, 1)))
#model.add(layers.Conv2D(100, (3, 1), activation='relu'))
#model.add(layers.MaxPooling2D((2, 1)))


model.add(layers.Flatten())
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(3, activation='softmax'))
model.summary()

#================================================

## For Pre-train model
#if pretrain==True:
#    save_path_full = os.path.join(save_path, model_name)
#    # load json and create model
#    json_file = open('model.json', 'r')
#    loaded_model_json = json_file.read()
#    json_file.close()
#    model = model_from_json(loaded_model_json)
#    # load weights into new model
#    model.load_weights(save_path_full)
#    print("Loaded model from disk")
##=======================================================
#=======================================================

model.compile(optimizer='adam',
              #loss='binary_crossentropy',   # Use one-hot encoding
              loss='sparse_categorical_crossentropy',  #Don't Use one-hot encoding
              metrics=['accuracy'])

save_path_full1 = os.path.join(save_path, top_model_name)
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint(save_path_full1, save_best_only=True, monitor='val_loss', mode='min')


history = model.fit(train_dataset, train_labels, batch_size=batch_size, epochs=epoch, verbose=1, callbacks=[earlyStopping, mcp_save],
                    validation_data=(test_dataset[0:1000,:], test_labels[0:1000,:]))
#history = model.fit(train_dataset, train_labels, batch_size=batch_size, epochs=epoch, verbose=1, callbacks=[earlyStopping, mcp_save],
#                    validation_split=0.25)




plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

save_path_full = os.path.join(save_path, model_name)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(save_path_full)
print("Saved model to disk")

score = model.evaluate(test_dataset, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pred = model.predict(test_dataset)
#pr_cls=model.predict_classes(test_dataset)
prediction = np.argmax(pred,1)
result=np.column_stack([test_labels, prediction])

print(confusion_matrix(test_labels, prediction, labels=[0, 1, 2]))
print(classification_report(test_labels, prediction, digits=3))

#print(confusion_matrix(test_labels, prediction, labels=[0, 1, 2]))
#print(classification_report(test_labels, prediction, digits=3))

#=============================================================
# For Compute Precision and Recall Curve
test_labels = to_categorical(test_labels).astype(np.float32)
##=====================================================================
#Pr1=[]
#Rc1=[]
#for i in range(n_classes):
#    Pr1, Rc1, _= precision_recall_curve(test_labels[:, i], pred[:, i], pos_label=1)
#    #Pr1.append(Pr11)
#    #Rc1.append(Rc11)
#AUC1=(Rc1,Pr1)
#print(AUC1)
#plt.figure()
#plt.step(Rc1, Pr1, where='post')
#
#plt.xlabel('Rec')
#plt.ylabel('Prec')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.0])
#
##======================================================================
#Pr=dict()
#Rc=dict()
#Pr["micro"], Rc["micro"], _ = precision_recall_curve(test_labels.ravel(), pred.ravel())

#Accuracy=[]
#Prec=[]
#Rec=[]
#Th=[0.0,0.0005,0.001,0.005,0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.81,0.825,0.84,0.85,0.86,0.875,0.885,0.895,0.9,0.91,0.92,0.925,0.93,0.94,0.941,0.942,0.943,0.944,0.945,0.946,0.947,0.948,0.949,0.95,0.955,0.956,0.957,0.958,0.959,0.96,0.961,0.962,0.963,0.964,0.965,0.967,0.968,0.969,0.97,0.971,0.972,0.973,0.974,0.975,0.976,0.977,0.978,0.979,0.98,0.981,0.982,0.983,0.984,0.985,0.986,0.987,0.988,0.989,0.99,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999,0.9991,0.9992,0.9993,0.9994,0.9995,0.9996,0.9997,0.9998,0.9999,1.0]
#for k in range(len(Th)):
#    Pr=[]
#    Rc=[]
#    Acc=[]
#    Avg_Pr=0
#    Avg_Rc=0
#    n_testsamp=len(test_labels)
#    for i in range(n_classes):
#        TP=0
#        TN=0
#        FP=0
#        FN=0
#        prd_lbl=[]
#        for j in range(n_testsamp):
#            if(pred[j,i]>Th[k]):
#                prd_lbl.append(1)
#                if test_labels[j,i]==1:
#                    TP+=1
#                else:
#                    FP+=1
#            else:
#                prd_lbl.append(0)
#                if test_labels[j,i]==0:
#                    TN+=1
#                else:
#                    FN+=1
#        #Pr, Rc, th= precision_recall_curve(test_labels[:, i], pred[:, i], pos_label=None)
#        print('**********************')
#        print('Performance for Class =',i)
#        #print('TP=FP=TN=FN=',TP,FP,TN,FN)
#        if TP == 0:
#            Pr.append(0)
#            Rc.append(0)
#        else:
#           Pr.append(TP/(TP+FP))
#           Rc.append(TP/(TP+FN))
#           Acc.append((TP+TN)/(TP+FP+TN+FN))
#    #Avg_Pr=Avg_Pr+Pr[i]
#    #Avg_Rc=Avg_Rc+Rc[i]
#       # print('Precision=%d\n Recall=%d\n',Pr, Rc)
#   
#    print('Avarage Precision=',sum(Pr)/n_classes, 'for Threshold',Th[k])
#    print('Avarage Recall=',sum(Rc)/n_classes,'for Threshold',Th[k])
#    print('Avarage Accuracy=',sum(Acc)/n_classes,'for Threshold',Th[k])
#    Prec.append((sum(Pr)/n_classes))
#    Rec.append((sum(Rc)/n_classes))
#    Accuracy.append((sum(Acc)/n_classes))
#    print('**********************')
#
#AUC=auc(Rec, Prec)
#print(AUC)
#plt.figure()
#plt.step(Rec, Prec, where='post')
#
#plt.xlabel('Rec')
#plt.ylabel('Prec')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.05])
#
#f = open('./results/Prec_70_60_80.pckl', 'wb')
#pickle.dump(Prec, f)
#f.close()
#f = open('./results/Rec_70_60_80.pckl', 'wb')
#pickle.dump(Rec, f)
#f.close()

#====================================================================

#wb = xlsxwriter.Workbook('./90_50_90/Results_paper.xlsx')
#ws = wb.add_worksheet('my sheet')
#ws.write_row(0, 0, ['Actual_class', 'Pred_class'])
#np.savetxt('./90_50_90/Results_paper.csv',result,delimiter=",")
#====================================================================

	



