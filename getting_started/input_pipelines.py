#Input functions must return the following two values 
#containing the final feature and label data to be fed into your model 
def my_input_fn():
	return feature_cols, labels


#Converting data to tensors

feature_colum_data = [1, 2.4, 0, 9.9, 3, 120]
feature_tensor = tf.constant(feature_column_data)

sparse_tensor = tf.SparseTensor(indices=[[0,1], [2,4]],values=[6, 0.5],dense_shape=[3, 5])

#passing input_fn data to the model
classifier.fit(input_fn=my_input_fn, steps=2000)

#parameterizing input functions
def my_input_function_training_set():
  return my_input_function(training_set)

classifier.fit(input_fn=my_input_fn_training_set, steps=2000)

#Using functools.partial
classifier.fit(input_fn=functools.partial(my_input_function,data_set=training_set), steps=2000)										  