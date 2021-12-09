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
file3 = pd.read_csv("Wet_t2.csv")
file4 = pd.read_csv("Dry_t2.csv")
filet = pd.concat([file3,file4],axis=0)
#filet.head()


# spilt in train and test
# measurement
mask_train_m = file["freqency(kHz)"].isin(["1","3","5","100"]) & file["area(mm2)"].isin(["0.0050","0.0049","0.0102","0.0026","0.0025"])
mask_test_m = file["freqency(kHz)"].isin(["10","50"]) 
mask_val_m = file["freqency(kHz)"].isin(["1","3","5","100"])  & file["area(mm2)"].isin(["0.0100"])
features_train_m = file[mask_train_m].drop(['Cm(pF)','Ct(pF)'], axis=1)
features_test_m = file[mask_test_m].drop(['Cm(pF)','Ct(pF)'], axis=1)
features_val_m = file[mask_val_m].drop(['Cm(pF)','Ct(pF)'], axis=1)
labels_train_m = file[mask_train_m].loc[:,['Cm(pF)','Ct(pF)']]
labels_test_m = file[mask_test_m].pop('Cm(pF)')
labels_val_m = file[mask_val_m].loc[:,['Cm(pF)','Ct(pF)']]
x_train_m = np.array(features_train_m)
x_test_m = np.array(features_test_m)
x_val_m = np.array(features_val_m)
y_train_m0 = np.array(labels_train_m).reshape(-1,2)
y_test_m = np.array(labels_test_m).reshape(-1,1)
y_val_m0 = np.array(labels_val_m).reshape(-1,2)

# theoretical
mask_train_t = filet["area(mm2)"].isin(["0.005","0.0049","0.0102","0.0026","0.0025"]) 
mask_val_t = filet["area(mm2)"].isin(["0.01"])
features_train_t = filet[mask_train_t].drop(['Cm(pF)','Ct(pF)'], axis=1)
features_val_t = filet[mask_val_t].drop(['Cm(pF)','Ct(pF)'], axis=1)
labels_train_t = filet[mask_train_t].loc[:,['Cm(pF)','Ct(pF)']]
labels_val_t = filet[mask_val_t].loc[:,['Cm(pF)','Ct(pF)']]
x_train_t = np.array(features_train_t)
x_val_t = np.array(features_val_t)
y_train_t0 = np.array(labels_train_t).reshape(-1,2)
y_val_t0 = np.array(labels_val_t).reshape(-1,2)

# all
x_train = np.concatenate((x_train_m,x_train_t),axis=0)
x_test = x_test_m
x_val = np.concatenate((x_val_m,x_val_t),axis=0)
y_val_m = np.concatenate((y_val_m0[:,0],y_val_t0[:,0]),axis=0).reshape(-1,1)
y_val_t = np.concatenate((y_val_m0[:,1],y_val_t0[:,1]),axis=0).reshape(-1,1)
y_train_m = np.concatenate((y_train_m0[:,0],y_train_t0[:,0]),axis=0).reshape(-1,1)
y_train_t = np.concatenate((y_train_m0[:,1],y_train_t0[:,1]),axis=0).reshape(-1,1)




print(x_train)
print(x_train.shape)
print(y_train_m)
print(y_train_m.shape)
print(y_train_t)
print(y_train_t.shape)
print(x_val)
print(x_val.shape)
print(y_val_m)
print(y_val_m.shape)
print(y_val_t)
print(y_val_t.shape)
print(x_test)
print(x_test.shape)
print(y_test_m)
print(y_test_m.shape)

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler3 = MinMaxScaler()

x_train=scaler1.fit_transform(x_train)
x_val=scaler1.transform(x_val)
x_test=scaler1.transform(x_test)
y_train_m=scaler2.fit_transform(y_train_m)
y_val_m=scaler2.transform(y_val_m)
y_train_t=scaler3.fit_transform(y_train_t)
y_val_t=scaler3.transform(y_val_t)



# custom loss function

def mse2(y_true,y_pred):
  mask = tf.math.logical_not(tf.math.equal(y_true, 0.0))
  loss = tf.math.square(y_pred - y_true)
  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask
  return tf.reduce_mean(loss, axis=-1)


# build model

input = tf.keras.Input(shape=(6,))
x = tf.keras.layers.Dense(units = 50, activation='relu')(input)
x1 = tf.keras.layers.Dense(units = 50, activation='relu')(x)
output_m = tf.keras.layers.Dense(units=1)(x1)
x2 = tf.keras.layers.Dense(units = 50, activation='relu')(x)
output_t = tf.keras.layers.Dense(units=1)(x2)
model = tf.keras.Model(inputs=input, outputs=[output_m,output_t])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=[mse2, mse2],loss_weights=[0.7,0.3],metrics=['mse', 'mse'])
callback =tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1,restore_best_weights=True)
hist = model.fit(x=x_train, y=[y_train_m, y_train_t],batch_size = 50,validation_data=(x_val,[y_val_m,y_val_t]),epochs=8000,shuffle=True,verbose=1,callbacks=[callback])
#checkpoint_path = "./checkpoint/number10/variables/variables"
#model.load_weights(checkpoint_path)


model.summary()



# unnormalization

y_predict_train=model.predict(x_train)
y_predict_train_m=y_predict_train[0]
y_predict_train_t=y_predict_train[1]
y_predict_val=model.predict(x_val)
y_predict_val_m=y_predict_val[0]
y_predict_val_t=y_predict_val[1]
y_predict_test=model.predict(x_test)
y_predict_test_m=y_predict_test[0]
y_predict_test_t=y_predict_test[1]
x_train=scaler1.inverse_transform(x_train)
x_val=scaler1.inverse_transform(x_val)
x_test=scaler1.inverse_transform(x_test)
y_train_m=scaler2.inverse_transform(y_train_m)
y_val_m=scaler2.inverse_transform(y_val_m)
y_predict_train_m=scaler2.inverse_transform(y_predict_train_m)
y_predict_val_m=scaler2.inverse_transform(y_predict_val_m)
y_predict_test_m=scaler2.inverse_transform(y_predict_test_m)
y_train_t=scaler3.inverse_transform(y_train_t)
y_val_t=scaler3.inverse_transform(y_val_t)
y_predict_train_t=scaler3.inverse_transform(y_predict_train_t)
y_predict_val_t=scaler3.inverse_transform(y_predict_val_t)
y_predict_test_t=scaler3.inverse_transform(y_predict_test_t)

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
print("r2_score in train data: %.4f" % r2_score(y_train_m[:4080], y_predict_train_m[:4080]))
print("mean squared error: %.4f" % mean_squared_error(y_train_m[:4080], y_predict_train_m[:4080]))
print("r2_score in val data: %.4f" % r2_score(y_val_m[:816], y_predict_val_m[:816]))
print("mean squared error: %.4f" % mean_squared_error(y_val_m[:816], y_predict_val_m[:816]))
print("r2_score in test data: %.4f" % r2_score(y_test_m, y_predict_test_m))
print("mean squared error: %.4f" % mean_squared_error(y_test_m, y_predict_test_m))



#acc

plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('accuracy in train data')

plt.plot(y_train_m[:4080],y_train_m[:4080], c='b',label='train data')
plt.scatter(y_train_m[:4080], y_predict_train_m[:4080], marker='+', label='predict data',s=25)

plt.xlabel('Capacitance_true (pF)')
plt.ylabel('Capacitance_predict (pF)')


plt.grid()
plt.legend()

plt.show()

plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('accuracy in validation data')

plt.plot(y_val_m[:816],y_val_m[:816], c='b',label='val data')
plt.scatter(y_val_m[:816], y_predict_val_m[:816], marker='+', label='predict data',s=25)

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


plt.scatter(x_train[:4080,5], y_train_m[:4080], c='red', marker='x', label='train data', s=25)
plt.scatter(x_train[:4080,5], y_predict_train_m[:4080], c='orange', label='predict data',s=15)

plt.xlabel('Voltage (V)')
plt.ylabel('Capacitance (pF)')

plt.xlim([-4.0,2.5])
plt.ylim([0,22])

plt.grid()
plt.legend()

plt.show()

plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('C-V curve with validation and predict data')


plt.scatter(x_val[:816,5], y_val_m[:816], c='red', marker='x', label='val data', s=25)
plt.scatter(x_val[:816,5], y_predict_val_m[:816], c='orange', label='predict data',s=15)

plt.xlabel('Voltage (V)')
plt.ylabel('Capacitance (pF)')

plt.xlim([-4.0,2.5])
plt.ylim([0,22])

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

df = pd.DataFrame (np.concatenate( (x_train[:4080], y_train_m[:4080], y_predict_train_m[:4080]),axis=1 ) )
filepath1= './csv/SSMTL_train_10.csv'
df.to_csv(filepath1,header=['frequency_kHz','area_mm2','RCA_STD','Dry_Wet','T21_T23','Voltage_V','Cm_pF','Cm_pred_pF'],index=False)

df = pd.DataFrame (np.concatenate( (x_val[:816], y_val_m[:816], y_predict_val_m[:816]),axis=1 ) )
filepath2= './csv/SSMTL_val_10.csv'
df.to_csv(filepath2,header=['frequency_kHz','area_mm2','RCA_STD','Dry_Wet','T21_T23','Voltage_V','Cm_pF','Cm_pred_pF'],index=False)

df = pd.DataFrame (np.concatenate( (x_test, y_test_m, y_predict_test_m),axis=1 ) )
filepath3= './csv/SSMTL_test_10.csv'
df.to_csv(filepath3,header=['frequency_kHz','area_mm2','RCA_STD','Dry_Wet','T21_T23','Voltage_V','Cm_pF','Cm_pred_pF'],index=False)

