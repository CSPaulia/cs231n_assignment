import numpy as np
import matplotlib.pyplot as plt
from vclab.classifiers.neural_net import TwoLayerNet, ThreeLayerNet

import logging
import os

logger = logging.getLogger('training_logger')
logger.setLevel(logging.INFO)

log_path = 'vclab-project1/project1/logs/training.log'
if os.path.exists(log_path):
    os.remove(log_path)

file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()

scores = net.loss(X)
logger.info('Your scores:')
logger.info(scores)
logger.info('correct scores:')
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
logger.info(correct_scores)

# The difference should be very small. We get < 1e-7
logger.info('Difference between your scores and correct scores:')
logger.info(np.sum(np.abs(scores - correct_scores)))


loss, _ = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133

# should be very small, we get < 1e-12
logger.info('Difference between your loss and correct loss:')
logger.info(np.sum(np.abs(loss - correct_loss)))


from vclab.gradient_check import eval_numerical_gradient

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss, grads = net.loss(X, y, reg=0.05)

# these should all be less than 1e-8 or so
for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    logger.info('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))


net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=False, logger=logger)

final_training_loss = stats['loss_history'][-1]
logger.info(f'Final training loss: {final_training_loss}')

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.savefig('vclab-project1/project1/figs/baby_net_training_loss.png')
plt.close()
from vclab.data_utils import load_CIFAR10


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'vclab-project1/project1/vclab/datasets/cifar-10-batches-py'

    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
    try:
        del X_train, y_train
        del X_test, y_test
        logger.info('Clear previously loaded data.')
    except:
        pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
logger.info(f'Train data shape: {X_train.shape}')
logger.info(f'Train labels shape: {y_train.shape}')
logger.info(f'Validation data shape: {X_val.shape}')
logger.info(f'Validation labels shape: {y_val.shape}')
logger.info(f'Test data shape: {X_test.shape}')
logger.info(f'Test labels shape: {y_test.shape}')

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True, logger=logger)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
logger.info(f'Validation accuracy: {val_acc}')

# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.subplots_adjust(hspace=0.5)
plt.savefig('vclab-project1/project1/figs/baseline_training_loss.png')
plt.close()

from vclab.vis_utils import visualize_grid
def show_net_weights(net, name):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.title(f'{name}')
    plt.gca().axis('off')
    plt.savefig(f'vclab-project1/project1/figs/{name}_weight.png')
    plt.close()

show_net_weights(net, 'baseline')

best_net = None # store the best model into this
#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

logger.info(f'training improved model---------------------------------')

input_size = 32 * 32 * 3
hidden_size = 256
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes, std=1e-4)

stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=2000, batch_size=200,
            learning_rate=2e-3, learning_rate_decay=0.95,
            reg=0.25, verbose=True, logger=logger)

best_net = net

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Print your validation accuracy: this should be above 48%
val_acc = (best_net.predict(X_val) == y_val).mean()
logger.info(f'Validation accuracy: {val_acc}')

# Visualize the weights of the best network
show_net_weights(best_net, 'improved')

# Print your test accuracy: this should be above 48%
test_acc = (best_net.predict(X_test) == y_test).mean()
logger.info(f'Test accuracy: {test_acc}')

# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.subplots_adjust(hspace=0.5)
plt.savefig('vclab-project1/project1/figs/improved_training_loss.png')
plt.close()

logger.info(f'training model with adjuested reg---------------------------------')

input_size = 32 * 32 * 3
hidden_size = 256
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes, std=1e-4)

stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=2000, batch_size=200,
            learning_rate=2e-3, learning_rate_decay=0.95,
            reg=0.75, verbose=True, logger=logger)

best_net = net

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Print your validation accuracy: this should be above 48%
val_acc = (best_net.predict(X_val) == y_val).mean()
logger.info(f'Validation accuracy: {val_acc}')

# Visualize the weights of the best network
show_net_weights(best_net, 'improved_reg')

# Print your test accuracy: this should be above 48%
test_acc = (best_net.predict(X_test) == y_test).mean()
logger.info(f'Test accuracy: {test_acc}')

# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.subplots_adjust(hspace=0.5)
plt.savefig('vclab-project1/project1/figs/improved_reg_training_loss.png')
plt.close()

logger.info(f'training model with cosine annealing---------------------------------')

input_size = 32 * 32 * 3
hidden_size = 256
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes, std=1e-4)

initial_lr = 2e-3
total_epochs = 5

loss_history = []
train_acc_history = []
val_acc_history = []
for epoch in range(1, 5):
    current_lr = 0.5 * (1 + np.cos(epoch / total_epochs * np.pi)) * initial_lr

    stats = net.train(X_train, y_train, X_val, y_val,
                num_iters=1000, batch_size=200,
                learning_rate=current_lr, learning_rate_decay=1.0,
                reg=0.25, verbose=False, logger=logger)
    
    loss_history += stats['loss_history']
    train_acc_history += stats['train_acc_history']
    val_acc_history += stats['val_acc_history']
    
    loss = stats['loss_history'][-1]
    logger.info(f'Step {epoch * 1000}, Learning Rate: {current_lr}, loss: {loss}')

best_net = net

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Print your validation accuracy: this should be above 48%
val_acc = (best_net.predict(X_val) == y_val).mean()
logger.info(f'Validation accuracy: {val_acc}')

# Visualize the weights of the best network
show_net_weights(best_net, 'coscine')

# Print your test accuracy: this should be above 48%
test_acc = (best_net.predict(X_test) == y_test).mean()
logger.info(f'Test accuracy: {test_acc}')

# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(loss_history)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(train_acc_history, label='train')
plt.plot(val_acc_history, label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.subplots_adjust(hspace=0.5)
plt.savefig('vclab-project1/project1/figs/cosine_annealing_training_loss.png')
plt.close()

logger.removeHandler(file_handler)
file_handler.close()