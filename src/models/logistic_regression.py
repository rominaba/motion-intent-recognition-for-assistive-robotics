from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

class LogisticRegressionModel:
    def __init__(self, max_iter: int = 10000, random_state: int = 42) -> None:
        self.weight_matrix = 0
        self.bias_vector = 0
        self.max_iter = max_iter
        self.random_state = random_state
    
    def softmax(self, z: ArrayLike) -> NDArray[np.floating[Any]]:
        z = np.asarray(z, dtype=float)
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def cross_entropy(
        self,
        probs: NDArray[np.floating[Any]],
        target_onehot: NDArray[np.floating[Any]],
    ) -> float:
        # probs: predicted class probabilities; target_onehot: one-hot true labels
        ce = 0.0
        for i in range(len(probs)):
            ce += target_onehot[i] * np.log(np.maximum(probs[i], 1e-15))
        return -ce

    def fit(
        self,
        X_train: NDArray[np.floating[Any]],
        y_train: NDArray[np.integer[Any]],
        learning_rate: float,
    ) -> None:
        # Initialize weights to small random numbers and biases to zero
        num_samples = X_train.shape[0]
        num_features = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        np_rng = np.random.default_rng(self.random_state)
        self.weight_matrix = np_rng.normal(loc=0.0, scale=0.01, size=(num_classes,num_features))
        self.bias_vector = np.zeros(num_classes)

        # One hot encoding for classes
        y_train_e = np.zeros((num_samples,num_classes))
        y_train_e[np.arange(num_samples), y_train] = 1

        yhat_train = np.zeros((num_samples,num_classes))
        previous_loss = -1
        flag_end = False
        num_iter = 0
        # Train until loss has small change or max number of iterations is reached
        while((not flag_end) and (num_iter<=self.max_iter)):
            num_iter += 1
            loss_sum = 0

            # Compute z, yhat, and cross entropy for each sample
            for i,x in enumerate(X_train):
                z = np.add(np.matmul(self.weight_matrix,x),self.bias_vector)
                yhat_train[i] = self.softmax(z)
                loss_sum += self.cross_entropy(yhat_train[i], y_train_e[i])

            # Average loss over the samples
            loss = loss_sum/num_samples

            # End training if loss is barely changing (converged)
            if(previous_loss != -1):
                if(abs(loss - previous_loss) <= 0.000001):
                    flag_end = True
            
            previous_loss = loss

            # Gradient calculations for W and b
            w_grad = (1/num_samples)*((yhat_train-y_train_e).T@X_train)
            b_grad = (1/num_samples)*(np.sum(yhat_train-y_train_e, axis=0))

            # Update weight matrix and bias based on gradient and learning rate
            self.weight_matrix -= learning_rate*w_grad
            self.bias_vector -= learning_rate*b_grad

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.integer[Any]]:
        """Return predicted class indices for input features X, without updating the model weights"""
        num_samples = X.shape[0]
        num_classes = self.weight_matrix.shape[0]
        yhat = np.zeros((num_samples, num_classes))
        for i, x in enumerate(X):
            z = np.add(np.matmul(self.weight_matrix, x), self.bias_vector)
            yhat[i] = self.softmax(z)
        return np.argmax(yhat, axis=1)