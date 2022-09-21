import numpy as np
import matplotlib.pyplot as plt
import argparse


def softmax(x):
    numer = np.exp(x - np.max(x, axis=1, keepdims=True))
    return numer / np.sum(numer, axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_initial_params(input_size, num_hidden, num_output):
    W1 = np.random.randn(input_size, num_hidden)
    b1 = np.zeros(num_hidden)
    W2 = np.random.randn(num_hidden, num_output)
    b2 = np.zeros(num_output)

    initial_params = {
        "b1": b1,
        "b2": b2,
        "W1": W1,
        "W2": W2
    }

    return initial_params


def forward_prop(data, labels, params):
    hidden = sigmoid((data @ params["W1"]) + params["b1"])
    outer = softmax((hidden @ params["W2"] + params["b2"]))
    cost = (1 / outer.shape[0]) * np.sum(
        -np.sum(labels * np.where(outer == 0, 0, np.log(outer)), axis=1), axis=0)
    return hidden, outer, cost


def backward_prop(data, labels, params, forward_prop_func):
    hidden, outer, cost = forward_prop_func(data, labels, params)
    deriv_z2 = (outer - labels) * (1 / outer.shape[0])
    deriv_z1 = deriv_z2 @ params["W2"].T * (1 - hidden) * hidden
    deriv_W1 = data.T @ deriv_z1
    deriv_W2 = hidden.T @ deriv_z2
    deriv_b1 = np.sum(deriv_z1, axis=0)
    deriv_b2 = np.sum(deriv_z2, axis=0)

    derivs = {
        "W1": deriv_W1,
        "W2": deriv_W2,
        "b1": deriv_b1,
        "b2": deriv_b2
    }
    return derivs


def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    hidden, outer, cost = forward_prop_func(data, labels, params)
    deriv_z2 = (outer - labels) * (1 / outer.shape[0])
    deriv_z1 = deriv_z2 @ params["W2"].T * (1 - hidden) * hidden
    deriv_W1 = ((2 * params["W1"]) * reg) + (data.T @ deriv_z1)
    deriv_W2 = ((2 * params["W2"]) * reg) + (hidden.T @ deriv_z2)
    deriv_b1 = np.sum(deriv_z1, axis=0)
    deriv_b2 = np.sum(deriv_z2, axis=0)

    grads = {
        "W1": deriv_W1,
        "W2": deriv_W2,
        "b1": deriv_b1,
        "b2": deriv_b2
    }
    return grads


def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func,
                           backward_prop_func):
    num_iters = int(train_data.shape[0] / batch_size)
    for i in range(num_iters):
        data = train_data[batch_size * i:batch_size * (i + 1)]
        labels = train_labels[i * batch_size:(i + 1) * batch_size]
        derivs = backward_prop_func(data, labels, params, forward_prop_func)
        params["W1"] = params["W1"] - (learning_rate * derivs["W1"])
        params["W2"] = params["W2"] - (learning_rate * derivs["W2"])
        params["b1"] = params["b1"] - (learning_rate * derivs["b1"])
        params["b2"] = params["b2"] - (learning_rate * derivs["b2"])

    return


def nn_train(
        train_data, train_labels, dev_data, dev_labels,
        get_initial_params_func, forward_prop_func, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):
    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels,
                               learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output, train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev


def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy


def compute_accuracy(output, labels):
    accuracy = (np.argmax(output, axis=1) ==
                np.argmax(labels, axis=1)).sum() * 1. / labels.shape[0]
    return accuracy


def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels


def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y


def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'],
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train, 'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train, 'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))

    return accuracy


def main(plot=True):
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    np.random.seed(100)
    train_data, train_labels = read_data('./trainBranAvg.csv', './labels_trainBranAvg.csv')
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(2000)
    train_data = train_data[p, :]
    train_labels = train_labels[p, :]

    dev_data = train_data[0:500, :]
    dev_labels = train_labels[0:500, :]
    train_data = train_data[500:, :]
    train_labels = train_labels[500:, :]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./testBranAvg.csv', './labels_testBranAvg.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }

    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels,
                             lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
                             args.num_epochs, plot)

    return baseline_acc, reg_acc


if __name__ == '__main__':
    main()