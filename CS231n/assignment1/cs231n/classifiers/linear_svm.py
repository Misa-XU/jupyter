from builtins import range
import numpy as np
from random import shuffle
#from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    scores = X @ W

    margin = np.maximum(scores - scores[np.arange(num_train), y].reshape(num_train, 1) + 1, 0)

    margin[np.arange(num_train), y] = 0

    loss = np.sum(margin) / num_train
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    margin[margin > 0] = 1
    margin[np.arange(num_train), y] = -np.sum(margin, axis=1)
    dW = X.T @ margin / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    scores = X @ W

    margin = np.maximum(scores - scores[np.arange(num_train), y].reshape(num_train, 1) + 1, 0)

    margin[np.arange(num_train), y] = 0

    loss = np.sum(margin) / num_train
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    margin[margin > 0] = 1
    margin[np.arange(num_train), y] = -np.sum(margin, axis=1)
    dW = X.T @ margin / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW
