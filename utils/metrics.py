__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"


import numpy as np
import matplotlib.pyplot as plt


def NMSE(prediction, ground_truth):
    i_hat = prediction[..., 0]
    i_true = ground_truth[..., 0]
    q_hat = prediction[..., 1]
    q_true = ground_truth[..., 1]

    MSE = np.mean(np.square(i_true - i_hat) + np.square(q_true - q_hat), axis=-1)
    energy = np.mean(np.square(i_true) + np.square(q_true), axis=-1)

    NMSE = np.mean(10 * np.log10(MSE / energy))
    return NMSE

def Accuracy(prediction, ground_truth):
    """Calculate classification accuracy and plot prediction comparison
    Args:
        prediction: numpy array of shape (batch, time) containing predicted class indices
        ground_truth: numpy array of shape (batch, time) containing true class indices
    Returns:
        accuracy: classification accuracy in percentage
    """
    # Calculate accuracy
    total = prediction.size
    correct = (prediction == ground_truth).sum()
    accuracy = 100 * correct / total
    

    
    return accuracy