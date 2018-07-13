import utils
import tensorflow as tf
import xlrd
import numpy as np
import os
import matplotlib.pyplot as plt

batch_size = 128
epochs = 30

#Create a parser to read in data and transform into a numpy array
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

#Since we created the data as tf.data.Dataset objects, we do not need to define the input and output data as placeholders and
#instead may iterate through them
train_data = tf.data.Dataset.from_tensor_slices(train)
test_data = tf.data.Dataset.from_tensor_slices(test)

#This just batches the data so that it is not all processed at once
train_data = train_data.batch(batch_size)
test_data = test_data.batch(batch_size)

#This iterates over the dataset and since we only want to create one iterator, we must use tf.data.Iterator.from_structure
#with both data_sets as params
iterator = tf.data.Iterator.from_structure(train_data, test_data)
img, label = iterator.get_next()

#This step initializes the iterator over both datasets separately
train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

#Create weights and biases
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('biases', initializer=tf.constant(0.0))

#Create an equation for the model
logits = tf.nn.softmax(train_data*w + b)

#Create a cross-entropy loss function since softmax was used
entropy = tf.nn.cross_entropy_with_logits(tain_data, logits, name='loss')
tf.reduce_mean(entropy) #To get our loss as a single value instead of a value for each iteration

#Create an optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01),minimize(loss)

#Train the model
with tf.Session() as sess:
    #Stor the graph in tensorboard
    writer = tf.FileWriter('./graphs', sess.graph)
    for i in range(epochs):
        sess.run(train_init)
        try:
            while True:
                sess.run(loss)
        except tf.errors.OutOfRangeError:
            pass
        
#Test the model
with tf.Session() as sess:
    for i in range(epochs):
        sess.run(test_init)
        try: 
            while True:
                sess.run(accuracy)
        except tf.errors.OutOfRangeError:
            pass


