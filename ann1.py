#import libraries
import numpy as np
import pandas as pd
import tensorflow as tf

#check tensorflow version
print("Tensorflow version: " + tf.__version__)

#data preprocessing
dataset = pd.read_csv("Churn_Modelling.csv") #read dataset
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:, -1].values

print("Raw data:")
print(X)
input('Press ENTER to continue...')
#print(y) #check data

#encoding categorical data
#gender
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

print("Encoding categorical data / gender:")
print(X)
input('Press ENTER to continue...')

#geography
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print("Encoding categorical data / country:")
print(X)
input('Press ENTER to continue...')

#splitting the dataset into the Training and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building the ANN
#initializing
ann = tf.keras.models.Sequential()

#Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Traning the ANN
#Compiling ANN --> defines the parameters relevant for training and calculate accuracy
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#Make a prediction
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
