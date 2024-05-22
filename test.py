# libraries
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from PIL import ImageTk, Image

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import load_model

#section 

skin_df = pd.read_csv("../HAM10000_metadata.csv")

#data analysis
skin_df.head()
skin_df.info()

sns.countplot(x = "dx", data = skin_df)
##preprocess  
data_folder_name = "HAM10000_images_part_1/"
ext =".jpg"

""" "HAM10000_images_part_1\ISIC_0027419.jpg"
data_folder_name + image_id[i] + .ext """


skin_df["path"] = [data_folder_name + i + ext for i in skin_df["image_id"]]
skin_df["image"] = skin_df["path"].map(lambda x: np.asarray(Image.open(x).resize(100,75)))
plt.imshow(skin_df["image"][0])

#neural network
#convert integer values
skin_df["dx_idx"] = pd.Categorical(skin_df["dx"]).codes
#save network data into
skin_df.to_pickle("skin_df.pkl")
#load pkl
skin_df = pd.read_pickle("skin_df.pkl")



#standarlization - compare 
x_train = np.asarray(skin_df["image"].to_list())
x_train_mean =  np.maen(x_train)
x_train_std = np.std(x_train)
x_train = (x_train_mean - x_train)/x_train_std


#one not encoding
y_train = to_categorical(skin_df["dx_idx"] ,num_classes = 7)


#CNN MODEL

input_shape = (75,100,3)
number_clasees = 7


##FEATURE EXT
model = Sequential()
model.add(Conv2D(32,kernel_size = (3,3) ,activation = "relu",padding = "Same",input_shape = input_shape))
model.add(Conv2D(32,kernel_size = (3,3) ,activation = "relu",padding = "Same"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

### 
model.add(Conv2D(64,kernel_size = (3,3) ,activation = "relu",padding = "Same"))
model.add(Conv2D(64,kernel_size = (3,3) ,activation = "relu",padding = "Same"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.5))

#CLASSIFICATION
model.add(Flatten())
model.add()(Dense(128,activation="relu"))
model.add(Dropout(0.5))
model.add()(Dense(number_clasees,activation="softmax"))
model.summary()

optimazer = Adam(lr = 0.001)
model.compile(optimazer = optimazer, loss = "categorical_crossentropy", metrics = ["accuarcy"])

### 
epochs = 5
batch_size = 25

history = model.fit(x = x_train, y = y_train,batch_size = batch_size,epochs = epochs, verbose = 1, shuffle = True)
model.save("my_model1.h5")

#load  to run the model
model1 =  load_model("my_model1.h5")
model2 =  load_model("my_model2.h5")


#predictions
index  =  5
y_pred = model1.predict(x_train[index].reshape(100,75,3))
y_pred_class = np.argmax(y_pred, axis=1)

# 

















