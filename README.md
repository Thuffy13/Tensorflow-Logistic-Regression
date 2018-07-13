# Tensorflow-Logistic-Regression
Logistic regression in tensorflow with the use of datasets and iterators forthe input data instead of placeholders
This logistic regression model was created with the use of datasets which is a class of tensorflow used in place of placeholders.
The iterator in this example is used to gothrough each epoch of the data and essentially replaces the need forany feed_dict commands.
The time need to train the model is significantly faster with this method since the data is no longer stored outside Tensorflow with the 
need to be added in, but is instead created as a new Tensorflow object which can be run directly on the Tensorflow API.
