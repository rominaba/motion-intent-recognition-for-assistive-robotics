import numpy as np

class Logistic_Regression_Model:
    def __init__(self, max_iter=10000, random_state=42):
        self.weight_matrix = 0
        self.bias_vector = 0
        self.max_iter = max_iter
        self.random_state = random_state
    
    def softmax(self,z):
        y = np.zeros(len(z))
        sum_z = np.sum(np.exp(z))
        for i,z_i in enumerate(z):
            y[i] = np.exp(z_i)/sum_z
        return y

    def cross_entropy(self,p,q):
        ce = 0
        for i in range(len(p)):
            ce += q[i]*np.log(p[i])
        ce = -ce
        return ce

    def fit(self, X_train, y_train, learning_rate):
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
                loss_sum += self.cross_entropy(yhat_train[i],y_train_e[i])

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

    def predict(self, X_test, y_test):
        num_samples = X_test.shape[0]
        num_classes = len(np.unique(y_test))

        # One hot encoding
        y_test_e = np.zeros((num_samples,num_classes))
        y_test_e[np.arange(num_samples), y_test] = 1

        yhat_test = np.zeros((num_samples,num_classes))
        # Compute z, and yhat for each sample
        for i,x in enumerate(X_test):
                z = np.add(np.matmul(self.weight_matrix,x),self.bias_vector)
                yhat_test[i] = self.softmax(z)

        # Choose most likely class (highest probability) for each sample
        yhat_test_class = np.argmax(yhat_test, axis=1)
        return yhat_test_class