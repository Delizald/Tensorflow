from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"

training_set = pd.read_csv("boston_train.csv", 
	skipinitialspace=True,
	skiprows=1, 
	names=COLUMNS)
test_set = pd.read_csv("boston_test.csv", 
	skipinitialspace=True,
	skiprows=1, 
	names=COLUMNS)
prediction_set = pd.read_csv("boston_predict.csv", 
	skipinitialspace=True, 
	skiprows=1,
	names=COLUMNS)

#creating the regressor
feature_cols = [tf.contrib.layers.real_calued_column(k)
	for k in FEATURES]

regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
	hidden_units=[10,10],
	model_dir="/tmp/boston_model")

#input_fn
def input_fn(data_set):
	feature_cols = {k: tf.constant(data_set[k].values)
		for k in FEATURES}
	labels = tf.constant(data_set[LABEL].values)
	return feature_cols,labels

#Training the regressor
regressor.fit(input_fn=lambda: input_fn(training_set),steps=5000)

#Evaluating the model
ev = regressor.evaluate(input_fn=lambda: input_fn(test_set),steps=1)

#retrieve the loss
loss_score = ev["loss"]
print("Loss: {0:f}".format(str(predictions))

