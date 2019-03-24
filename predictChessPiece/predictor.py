import math
import numpy as np
import tensorflow as tf
import base64
from PIL import Image
from io import BytesIO

filename="predictChessPiece/db/trained_weights.npz"

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
       
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    with tf.Session() as sess:
        prediction = sess.run(p, feed_dict = {x: X})
    
    return prediction

def convertToNumpyArray(encodedImage):
    """
    # Converts an encoded image into a numpy array array of Bytes
    """
    b64img = base64.b64decode(encodedImage)
    img = Image.open(BytesIO(b64img)).convert("RGB")
    
    return np.array(img)
        

def loadParams():
    """
    loads the trained weights from a file to use in a prediction
    """
    params = np.load(filename)
    parameters = params['arr_0'].item()

    return parameters

def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
                  
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    # Numpy equivalent: Z1 = np.dot(W1, X) + b1
    Z1 = tf.add(tf.matmul(W1,X),b1)
    # Numpy equivalent: A1 = relu(Z1)
    A1 = tf.nn.relu(Z1)
    # Numpy equivalent: Z2 = np.dot(W2, A1) + b2
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    # Numpy equivalent: A2 = relu(Z2)
    A2 = tf.nn.relu(Z2)
    # Numpy equivalent: Z3 = np.dot(W3, A2) + b3
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    
    return Z3
            