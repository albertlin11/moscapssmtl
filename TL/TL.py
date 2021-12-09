#!/usr/bin/env python
# coding: utf-8


from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os
import random

seed_value= 1
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

plt.close("all")

file1 = pd.read_csv("Wet_m1.csv")
file2 = pd.read_csv("Dry_m1.csv")
file = pd.concat([file1,file2],axis=0)
#file.head()


# spilt in train and test

mask_train = file["freqency(kHz)"].isin(["1","100"]) & file["area(mm2)"].isin(["0.005","0.0049","0.0102","0.0026","0.0025"])
mask_test = file["freqency(kHz)"].isin(["3","5","10","50"]) 
mask_val = file["freqency(kHz)"].isin(["1","100"])  & file["area(mm2)"].isin(["0.01"])
features_train = file[mask_train].drop(['C(pF)'], axis=1)
features_test = file[mask_test].drop(['C(pF)'], axis=1)
features_val = file[mask_val].drop(['C(pF)'], axis=1)
labels_train = file[mask_train].pop('C(pF)')
labels_test = file[mask_test].pop('C(pF)')
labels_val = file[mask_val].pop('C(pF)')
x_train = np.array(features_train)
x_test = np.array(features_test)
x_val = np.array(features_val)
y_train_m = np.array(labels_train).reshape(-1,1)
y_test_m = np.array(labels_test).reshape(-1,1)
y_val_m = np.array(labels_val).reshape(-1,1)

print(x_train)
print(x_train.shape)
print(y_train_m)
print(y_train_m.shape)
print(x_test)
print(x_test.shape)
print(y_test_m)
print(y_test_m.shape)
print(x_val)
print(x_val.shape)
print(y_val_m)
print(y_val_m.shape)

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

x_train=scaler1.fit_transform(x_train)
x_test=scaler1.transform(x_test)
x_val=scaler1.transform(x_val)
y_train_m=scaler2.fit_transform(y_train_m)
y_val_m=scaler2.transform(y_val_m)



# build model

base_model = tf.keras.models.load_model('./model/base50.h5')
base_model.trainable = False
base_model.summary()

base_layer = base_model.get_layer(index=1)
base_layer.trainable = False

x = tf.keras.Input(shape=(6,))
x1=base_layer(x)
x2=tf.keras.layers.Dense(units = 10, activation='relu',name='dense_1')(x1)
y1=tf.keras.layers.Dense(units =  1,name='dense_2')(x2)
model = tf.keras.Model(inputs=x, outputs=y1, name='b_model')

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
hist =model.fit(x_train, y_train_m,batch_size = 50,validation_data=(x_val,y_val_m),epochs = 30,shuffle=True,verbose = 1)
#checkpoint_path = "./checkpoint/number10/variables/variables"
#model.load_weights(checkpoint_path)



model.summary()



# unnormalization

y_predict_train_m=model.predict(x_train)
y_predict_train_m=scaler2.inverse_transform(y_predict_train_m)
y_predict_val_m=model.predict(x_val)
y_predict_val_m=scaler2.inverse_transform(y_predict_val_m)
y_predict_test_m=model.predict(x_test)
y_predict_test_m=scaler2.inverse_transform(y_predict_test_m)
y_train_m=scaler2.inverse_transform(y_train_m)
y_val_m=scaler2.inverse_transform(y_val_m)
x_train=scaler1.inverse_transform(x_train)
x_val=scaler1.inverse_transform(x_val)
x_test=scaler1.inverse_transform(x_test)

print(x_train)
print(y_train_m)
print(y_predict_train_m)
print(x_val)
print(y_val_m)
print(y_predict_val_m)
print(x_test)
print(y_test_m)
print(y_predict_test_m)



# predict

from sklearn.metrics import mean_squared_error, r2_score
# 模型越好：r2→1 / 模型越差：r2→0
print("r2_score in train data: %.4f" % r2_score(y_train_m, y_predict_train_m))
print("mean squared error: %.4f" % mean_squared_error(y_train_m, y_predict_train_m))
print("r2_score in val data: %.4f" % r2_score(y_val_m, y_predict_val_m))
print("mean squared error: %.4f" % mean_squared_error(y_val_m, y_predict_val_m))
print("r2_score in test data: %.4f" % r2_score(y_test_m, y_predict_test_m))
print("mean squared error: %.4f" % mean_squared_error(y_test_m, y_predict_test_m))



#acc

plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('accuracy in train data')

plt.plot(y_train_m,y_train_m, c='b',label='train data')
plt.scatter(y_train_m, y_predict_train_m, marker='+', label='predict data',s=25)

plt.xlabel('Capacitance_true (pF)')
plt.ylabel('Capacitance_predict (pF)')


plt.grid()
plt.legend()

plt.show()

plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('accuracy in validation data')

plt.plot(y_val_m,y_val_m, c='b',label='val data')
plt.scatter(y_val_m, y_predict_val_m, marker='+', label='predict data',s=25)

plt.xlabel('Capacitance_true (pF)')
plt.ylabel('Capacitance_predict (pF)')


plt.grid()
plt.legend()

plt.show()

plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('accuracy in test data')

plt.plot(y_test_m,y_test_m, c='b',label='test data')
plt.scatter(y_test_m,y_predict_test_m, marker='+', label='predict data',s=25)

plt.xlabel('Capacitance_true (pF)')
plt.ylabel('Capacitance_predict (pF)')


plt.grid()
plt.legend()

plt.show()



#cv

plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('C-V curve with train and predict data')


plt.scatter(x_train[:,5], y_train_m, c='red', marker='x', label='train data', s=25)
plt.scatter(x_train[:,5], y_predict_train_m, c='orange', label='predict data',s=15)

plt.xlabel('Voltage (V)')
plt.ylabel('Capacitance (pF)')

plt.xlim([-4.0,2.5])

plt.grid()
plt.legend()

plt.show()

plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('C-V curve with validation and predict data')


plt.scatter(x_val[:,5], y_val_m, c='red', marker='x', label='val data', s=25)
plt.scatter(x_val[:,5], y_predict_val_m, c='orange', label='predict data',s=15)

plt.xlabel('Voltage (V)')
plt.ylabel('Capacitance (pF)')

plt.xlim([-4.0,2.5])

plt.grid()
plt.legend()

plt.show()

plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('C-V curve with test and predict data')


plt.scatter(x_test[:,5], y_test_m, c='red', marker='x', label='test data', s=25)
plt.scatter(x_test[:,5], y_predict_test_m, c='orange', label='predict data',s=15)

plt.xlabel('Voltage (V)')
plt.ylabel('Capacitance (pF)')

plt.xlim([-4.0,2.5])

plt.grid()
plt.legend()

plt.show()



print(hist.history.keys())



# summarize history for loss
plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='val')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()



# csv

df = pd.DataFrame (np.concatenate( (x_train, y_train_m, y_predict_train_m),axis=1 ) )
filepath1= './csv/TL_train_5010.csv'
df.to_csv(filepath1,header=['frequency_kHz','area_mm2','RCA_STD','Dry_Wet','T21_T23','Voltage_V','Cm_pF','Cm_pred_pF'],index=False)

df = pd.DataFrame (np.concatenate( (x_val, y_val_m, y_predict_val_m),axis=1 ) )
filepath2= './csv/TL_val_5010.csv'
df.to_csv(filepath2,header=['frequency_kHz','area_mm2','RCA_STD','Dry_Wet','T21_T23','Voltage_V','Cm_pF','Cm_pred_pF'],index=False)

df = pd.DataFrame (np.concatenate( (x_test, y_test_m, y_predict_test_m),axis=1 ) )
filepath3= './csv/TL_test_5010.csv'
df.to_csv(filepath3,header=['frequency_kHz','area_mm2','RCA_STD','Dry_Wet','T21_T23','Voltage_V','Cm_pF','Cm_pred_pF'],index=False)

