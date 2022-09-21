import numpy as np
import util
import math


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    clf = GDA(theta_0=np.array([[1, 0, 0], [1, 0, 0]], dtype=float))
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_valid)
    np.savetxt(save_path, predictions)
    clf.theta[0] = clf.theta[0] + clf.theta[1]
    util.plot(x_valid, y_valid, clf.theta[0], 'kachowmybrother')
    # *** END CODE HERE ***


class KNN:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a KNN model to training set given by x and y

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        # DONE: mu_1, ..., mu_10
        # TODO: Phi, Sigma
        y = y.reshape(len(y), 1)
        """
        phi = 0
        for obj in y:
            phi += obj
        phi = phi / len(y)
        """
        phi = 1 / 10

        mu_1_num = np.zeros(shape=(1, 12))
        mu_1_denom = 0
        mu_2_num = np.zeros(shape=(1, 12))
        mu_2_denom = 0
        mu_3_num = np.zeros(shape=(1, 12))
        mu_3_denom = 0
        mu_4_num = np.zeros(shape=(1, 12))
        mu_4_denom = 0
        mu_5_num = np.zeros(shape=(1, 12))
        mu_5_denom = 0
        mu_6_num = np.zeros(shape=(1, 12))
        mu_6_denom = 0
        mu_7_num = np.zeros(shape=(1, 12))
        mu_7_denom = 0
        mu_8_num = np.zeros(shape=(1, 12))
        mu_8_denom = 0
        mu_9_num = np.zeros(shape=(1, 12))
        mu_9_denom = 0
        mu_10_num = np.zeros(shape=(1, 12))
        mu_10_denom = 0

        for i in range(len(y)):
            if y[i] == 1:
                mu_1_num = np.add(mu_1_num, x[i])
                mu_1_denom += 1
            elif y[i] == 2:
                mu_2_num = np.add(mu_2_num, x[i])
                mu_2_denom += 1
            elif y[i] == 3:
                mu_3_num = np.add(mu_3_num, x[i])
                mu_3_denom += 1
            elif y[i] == 4:
                mu_4_num = np.add(mu_4_num, x[i])
                mu_4_denom += 1
            elif y[i] == 5:
                mu_5_num = np.add(mu_5_num, x[i])
                mu_5_denom += 1
            elif y[i] == 6:
                mu_6_num = np.add(mu_6_num, x[i])
                mu_6_denom += 1
            elif y[i] == 7:
                mu_7_num = np.add(mu_7_num, x[i])
                mu_7_denom += 1
            elif y[i] == 8:
                mu_8_num = np.add(mu_8_num, x[i])
                mu_8_denom += 1
            elif y[i] == 9:
                mu_9_num = np.add(mu_9_num, x[i])
                mu_9_denom += 1
            else:
                mu_10_num = np.add(mu_10_num, x[i])
                mu_10_denom += 1
        mu_1 = mu_1_num / mu_1_denom
        mu_2 = mu_2_num / mu_2_denom
        mu_3 = mu_3_num / mu_3_denom
        mu_4 = mu_4_num / mu_4_denom
        mu_5 = mu_5_num / mu_5_denom
        mu_6 = mu_6_num / mu_6_denom
        mu_7 = mu_7_num / mu_7_denom
        mu_8 = mu_8_num / mu_8_denom
        mu_9 = mu_9_num / mu_9_denom
        mu_10 = mu_10_num / mu_10_denom

        sigma = np.zeros(shape=(12, 12))
        for i in range(len(y)):
            cur = x[i]
            cur = cur.reshape(1, 12)
            if y[i] == 1:
                cur -= mu_1
            elif y[i] == 2:
                cur -= mu_2
            elif y[i] == 3:
                cur -= mu_3
            elif y[i] == 4:
                cur -= mu_4
            elif y[i] == 5:
                cur -= mu_5
            elif y[i] == 6:
                cur -= mu_6
            elif y[i] == 7:
                cur -= mu_7
            elif y[i] == 8:
                cur -= mu_8
            elif y[i] == 9:
                cur -= mu_9
            else:
                cur -= mu_10
            sigma = np.add(sigma, cur.T @ cur)
        sigma = sigma / len(y)
        mu_1 = mu_1.reshape(1, 12)
        mu_2 = mu_2.reshape(1, 12)
        mu_3 = mu_3.reshape(1, 12)
        mu_4 = mu_4.reshape(1, 12)
        mu_5 = mu_5.reshape(1, 12)
        mu_6 = mu_6.reshape(1, 12)
        mu_7 = mu_7.reshape(1, 12)
        mu_8 = mu_8.reshape(1, 12)
        mu_9 = mu_9.reshape(1, 12)
        mu_10 = mu_10.reshape(1, 12)
        # THETA NEEDS TO BE SHAPE (12, 1)
        """
           sigma_inv = np.linalg.inv(sigma)
           theta_denom_builder = 0  # Note: possibly -.1 instead of -.5? Something to keep note of
           theta_denom_builder += np.e ** (-.5 * (x-mu_1).T @ sigma_inv @ (x-mu_1)) * phi
           theta_denom_builder += np.e ** (-.5 * (x-mu_2).T @ sigma_inv @ (x-mu_2)) * phi
           theta_denom_builder += np.e ** (-.5 * (x-mu_3).T @ sigma_inv @ (x-mu_3)) * phi
           theta_denom_builder += np.e ** (-.5 * (x-mu_4).T @ sigma_inv @ (x-mu_4)) * phi
           theta_denom_builder += np.e ** (-.5 * (x-mu_5).T @ sigma_inv @ (x-mu_5)) * phi
           theta_denom_builder += np.e ** (-.5 * (x-mu_6).T @ sigma_inv @ (x-mu_6)) * phi
           theta_denom_builder += np.e ** (-.5 * (x-mu_7).T @ sigma_inv @ (x-mu_7)) * phi
           theta_denom_builder += np.e ** (-.5 * (x-mu_8).T @ sigma_inv @ (x-mu_8)) * phi
           theta_denom_builder += np.e ** (-.5 * (x-mu_9).T @ sigma_inv @ (x-mu_9)) * phi
           theta_denom_builder += np.e ** (-.5 * (x-mu_10).T @ sigma_inv @ (x-mu_10)) * phi

           # placeholder_theta = (1/2)*(2*mu_1 - 2*mu_0) @ np.linalg.inv(sigma) + np.log((1-phi)/phi)
           # placeholder_theta0 = 1/2 * ((mu_0 @ np.linalg.inv(sigma) @ mu_0.T) - (mu_1 @ np.linalg.inv(sigma) @ mu_1.T)) + np.log((1-phi)/phi)

           # np.copyto(self.theta[0, 1:], placeholder_theta)
           # np.copyto(self.theta[1, 1:], np.array(placeholder_theta0))
        """
        return phi, sigma, mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7, mu_8, mu_9, mu_10
        # *** END CODE HERE ***

    def predict(self, x_test, x_train, y_train):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        CLASSIFY BY NEAREST NEIGHBOR
        """
        # *** START CODE HERE ***
        retvals = np.zeros(len(x_test))
        for i in range(len(x_test)):
            min_dist = 100000000
            min_d_ind = -1
            for j in range(len(x_train)):
                cur_dist = math.dist(x_test[i], x_train[j])
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    min_d_ind = y_train[j]
            retvals[i] = min_d_ind

        return retvals
        # *** END CODE HERE