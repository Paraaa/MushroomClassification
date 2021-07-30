import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.metrics import accuracy
from plotter import Plotter

plotter = Plotter(save_plots=False)
mushroom_dataset = pd.read_csv("./Dataset/mushrooms.csv")

#drop data
mushroom_dataset = mushroom_dataset.drop(['veil-type'], axis=1)


#convert dataset to numerical values
for column in mushroom_dataset.columns:
    for index, element in enumerate(mushroom_dataset[column].unique()):
        mushroom_dataset[column] = mushroom_dataset[column].replace(element, index)

#create train and test sets
train_dataset = mushroom_dataset.sample(frac=0.8, random_state=42)
test_dataset  = mushroom_dataset.drop(train_dataset.index)
#create features for test and train set
train_features = train_dataset.copy()
test_features  = test_dataset.copy()
#create labels for test and train set
train_labels   = train_features.pop("class")
test_labels    = test_features.pop("class")


#build model
model = tf.keras.Sequential([
    layers.Dense(units=train_features.shape[1]),
    layers.Dense(units=1,activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_features, train_labels,validation_split=0.2, epochs=80, batch_size=64)
test_loss, test_accuracy = model.evaluate(test_features, test_labels, verbose=2)

print("Test loss: ", test_loss)
print("Test accuracy: ", test_accuracy)


plotter.lineplot({"Accuracy": history.history["accuracy"], "Val accuracy": history.history["val_accuracy"]}, title="Accuracy")
plotter.lineplot({"Loss":history.history["loss"], "Val loss": history.history["val_loss"]}, title="Loss")
plotter.show()