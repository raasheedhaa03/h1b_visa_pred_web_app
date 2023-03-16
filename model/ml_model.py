import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping,History
from keras import backend as K
from keras.models import Model
from sklearn.model_selection import train_test_split

df = pd.read_csv('basedataset.csv',encoding='latin')
df.drop('WORKSITE_STATE_FULL',inplace=True,axis=1)
df.drop('CASE_NUMBER',inplace=True,axis=1)
df.drop('YEAR',inplace=True,axis=1)
df.drop('WORKSITE_CITY',inplace=True,axis=1)
df.drop('WORKSITE_STATE_ABB',inplace=True,axis=1)
df.drop('SOC_CODE',inplace=True,axis=1)
df.drop(df.index[(df["CASE_STATUS"] == "WITHDRAWN")],axis=0,inplace=True)
df.drop(df.index[(df["CASE_STATUS"] == "CERTIFIED-WITHDRAWN")],axis=0,inplace=True)
dff=df.dropna()

Q1 = np.percentile(dff['PREVAILING_WAGE'] , 25)
Q3 = np.percentile(dff['PREVAILING_WAGE'] , 75)
IQR = Q3 - Q1
ul = Q3+1.5*IQR
ll = Q1-1.5*IQR
dff["PREVAILING_WAGE"] = np.where(dff["PREVAILING_WAGE"]  >ul, ul,dff["PREVAILING_WAGE"] )
dff["PREVAILING_WAGE"] = np.where(dff["PREVAILING_WAGE"]  <ll, ll,dff["PREVAILING_WAGE"] )
dff.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

sdf=pd.DataFrame(dff[dff['CASE_STATUS'] == 'CERTIFIED'])
rdf=pd.DataFrame(dff[dff['CASE_STATUS']=='DENIED'])
fdf=rdf.groupby('FULL_TIME_POSITION', group_keys=False).apply(lambda x: x.sample(382544,replace=True))
cdf= pd.concat([fdf, sdf], ignore_index=True)

cdf['CASE_STATUS'].replace(['CERTIFIED', 'DENIED'],[1, 0], inplace=True)
cdf['FULL_TIME_POSITION'].replace(['Y', 'N'],[1, 0], inplace=True)

Y=pd.DataFrame(cdf['CASE_STATUS'])
cdf.drop('CASE_STATUS',inplace=True,axis=1)

minmax_scaler = MinMaxScaler()
model1= minmax_scaler.fit(cdf[['PREVAILING_WAGE']])
cdf['PREVAILING_WAGE']=model1.transform(cdf[['PREVAILING_WAGE']])

encoder= ce.BinaryEncoder(cols=['EMPLOYER_NAME','SOC_NAME','JOB_TITLE','WORKSITE'],return_df=True)
model2=encoder.fit(cdf)
nndf=model2.transform(cdf)



nndf_train,nndf_test,Y_train,Y_test=train_test_split(nndf,Y,test_size=0.2)

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1024,activation='relu'))
model.add(keras.layers.Dense(units=512, activation='relu'))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=32, activation='relu'))
model.add(keras.layers.Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(nndf_train, Y_train,batch_size=450, validation_data=(nndf_test,Y_test),epochs=100,callbacks=EarlyStopping(monitor='val_loss',verbose=1,min_delta=0.0001,patience=60))

pickle.dump(model,open("ml_model.sav","wb"))
pickle.dump(model1,open("scaler.sav","wb"))
pickle.dump(model2,open("encoder.sav","wb"))
