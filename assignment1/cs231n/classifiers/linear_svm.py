from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,y[i]] -=X[i] #+ 2*reg*W[:,y[i]]
                dW[:,j]+=X[i] #+ 2*reg*W[:,j]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW/=num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2* reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    y_pick = y.T
    y_pick = y_pick.reshape((1,X.shape[0]))
    y_pick = y.tolist()
    correct_class_score = np.choose(y_pick, scores.T)
    correct_class_score = np.reshape(correct_class_score,(correct_class_score.shape[0],1))
    #gate = scores - correct_class_score
    # 2 mask for updates
    #other_mask = gate > 0
    #correct_mask = ~gate
    #sum_x = np.sum(X.T,axis=1)
    #print('dim of sum_x is',sum_x.shape)
    #dW[: = correct_mask*
    other_class = scores - correct_class_score
    other_class_mask = ~(other_class == 0)
    margins = np.maximum(0, scores - correct_class_score + 1)
    margins[np.arange(X.shape[0]), y] = 0
    sum_margin = np.sum(margins,axis=1)
    #print('shape of sum_margin is',sum_margin.shape)    
    loss = np.sum(sum_margin,axis=0)/X.shape[0]
    loss += reg * np.sum(W * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    xtrans = X.T
    gate = 1.0*(margins > 0)
    #print(gate)
    sum_in_row = np.sum(gate, axis=1)
    #print(sum_in_row)
    #gate*= np.reshape(sum_in_row,(sum_in_row.shape[0],1))
    #print(sum_in_row)
    gate[np.arange(X.shape[0]), y] = -1*sum_in_row
    #print(gate)
    dW = np.dot(xtrans, gate)
    #print('dW is',dW.shape)
    dW /=X.shape[0]
    dW += 2*reg * W
    #Add reg
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
