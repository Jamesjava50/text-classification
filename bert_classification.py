#!/usr/bin/env python
# coding: utf-8




import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
df = pd.read_csv("Testy.csv")
df.head(10)





df['Topic'].value_counts()
df_Appointments = df[df['Topic']=='Appointments']
df_public_holidays = df[df['Topic']=='public_holidays']
df_Job_vacancies = df[df['Topic']=='Job_vacancies']
df_Tender = df[df['Topic']=='Tender']
df_balanced = pd.concat([df_Appointments, df_public_holidays, df_Tender, df_Job_vacancies])
df_balanced['Topic'].value_counts()





df_balanced['Appointments']=df_balanced['Topic'].apply(lambda x: 1 if x=='Appointments' else 2 if x=='Tender' else 3 if x=='public_holidays' else 0) 
df_balanced.sample(20)





df.shape





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_balanced['Task'],df_balanced['Appointments'], stratify=df_balanced['Appointments'])





bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")



text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)





l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='softmax', name="output")(l)





model = tf.keras.Model(inputs=[text_input], outputs = [l])
model.summary()





METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
 loss='categorical_crossentropy',
 metrics=METRICS)





model.fit(X_train, y_train, epochs=10)
y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()





y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()





import numpy as np

y_predicted = np.where(y_predicted == 1, 2, 3)
y_predicted





sample_dataset = ["IN EXERCISE of the authority conferred in me by the Constitution of Kenya, the County Governments Act and the Public Appointments (County Assemblies Approval) Act, I, appoint"]
model.predict(sample_dataset)





