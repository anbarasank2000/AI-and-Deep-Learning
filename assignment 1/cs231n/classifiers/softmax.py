from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    

          
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # sum_of_exp = 0.0
    num_classes = W.shape[1]
    num_train = X.shape[0]
    # itereating through the dataset
    for i in range(num_train):
      # scores by dot product of feautures and weights
      scores = X[i].dot(W)
      exp_scores = np.exp(scores)
      exp_correct_class_score = np.exp(scores[y[i]])
      sum_of_exp = np.sum(exp_scores)
      # calculating the probability of classes
      score_probablity = exp_correct_class_score/sum_of_exp
      for j in range(num_classes):
        # gradient for incorrect classes
        dW[:,j] += X[i]*(np.exp(scores[j]))/sum_of_exp
      # softmax loss
      loss += -np.log(score_probablity)
      # subtracting the row corresponding to the correct class
      dW[:,y[i]] -= X[i]
    # taking average
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * (2 * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    # dot product to get scores
    scores = X.dot(W)
    # getting neccessary exponents
    exp_scores = np.exp(scores)
    exp_correct_class_score = np.exp(scores)
    sum_of_exp = np.sum(exp_scores, axis=1, keepdims=True)
    score_probablity = exp_correct_class_score/sum_of_exp
    # average loss of all dataset
    loss = np.sum(-np.log(score_probablity[np.arange(num_train),y]))
    loss /= num_train
    # adding the regularization term
    loss += reg * np.sum(W * W)
    # correct class gradient
    grad = np.zeros(score_probablity.shape)
    grad[np.arange(num_train),y] = 1
    # updating the scores by subtracting 1 from correct class
    scores = score_probablity - grad
    # average gradient of all dataset
    dW = np.dot(X.T, scores)
    dW /= num_train
    # including the regularizaion
    dW += reg * (2 * W)



    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
