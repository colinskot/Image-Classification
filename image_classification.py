from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
from sklearn.preprocessing import normalize
import problem_unittests as tests
import tarfile
import helper
import numpy as np
import pickle
import tensorflow as tf
import random


cifar10_dataset_folder_path = 'cifar-10-batches-py'
tar_gz_path = 'cifar-10-python.tar.gz'
save_model_path = './image_classification'

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_path,
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open(tar_gz_path) as tar:
        tar.extractall()
        tar.close()


def display_data(batch_id=1, sample_id=6):
    """
    Display a picture from a batch + sample with its information
    :param batch_id: batch id number
    :param sample_id: sample id number
    """
    helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    :param x: List of image data.  The image shape is (32, 32, 3)
    :return: Numpy array of normalize data
    """
    # 8-bit image range [0-255], so x/255 puts image data values at range [0, 1]
    return x/255


def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # return diagonal array rows in position x.reshape(-1)
    arr = np.array(x).reshape(-1)
    one_hot = np.eye(10, dtype=int)[arr]
    return one_hot


def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # return placeholder with shape image_shape and name x
    return tf.placeholder(tf.float32, shape=(None, *image_shape), name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # return placeholder with shape n_classes and name y
    return tf.placeholder(tf.float32, shape=(None, n_classes), name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # return placeholder with name keep_prob
    return tf.placeholder(tf.float32, name='keep_prob')


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """

    # weights and biases
    weights = tf.Variable(tf.truncated_normal([*conv_ksize, int(x_tensor.shape[3]), conv_num_outputs], stddev=0.1))
    biases = tf.Variable(tf.zeros(conv_num_outputs))

    # convolutional layer, add biases and activate
    conv_layer = tf.nn.conv2d(x_tensor, weights, strides=[1, *conv_strides, 1], padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, biases)
    conv_layer = tf.nn.relu6(conv_layer)

    # pooling layer
    conv_layer = tf.nn.max_pool(conv_layer, [1, *pool_ksize, 1], [1, *pool_strides, 1], padding='SAME')

    # return graph
    return conv_layer


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """

    # returns flattened layer
    return tf.contrib.layers.flatten(x_tensor)


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """

    # create weights & biases
    weights = tf.Variable(tf.truncated_normal([int(x_tensor.shape[1]), num_outputs], stddev=0.1))
    biases  = tf.Variable(tf.zeros(num_outputs))

    # multiply weights & add biases
    out = tf.matmul(x_tensor, weights)
    out = tf.add(out, biases)

    # activation function: relu6
    out = tf.nn.relu6(out)

    return out


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """

    # create weights & biases
    weights = tf.Variable(tf.truncated_normal([int(x_tensor.shape[1]), num_outputs], stddev=0.1))
    biases  = tf.Variable(tf.zeros(num_outputs))

    # multiply weights & add biases
    out = tf.matmul(x_tensor, weights)
    out = tf.add(out, biases)

    return out


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """

    x_tensor = x
    conv_num_outputs = [64, 128, 256]
    conv_ksize = [2, 2]
    conv_strides = [2, 2]
    pool_ksize = [2, 2]
    pool_strides = [2, 2]
    fc_num_outputs = 100

    # Convolutional Layers
    conv1 = conv2d_maxpool(x_tensor, conv_num_outputs[0], conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv2 = conv2d_maxpool(conv1, conv_num_outputs[1], conv_ksize, conv_strides, pool_ksize, pool_strides)
    drop1 = tf.nn.dropout(conv2, keep_prob)
    conv3 = conv2d_maxpool(drop1, conv_num_outputs[2], conv_ksize, conv_strides, pool_ksize, pool_strides)

    # Flatten Layer
    flat = flatten(conv3)

    # Fully Connected Layers
    fc1 = fully_conn(flat, fc_num_outputs)
    fc2 = fully_conn(fc1, fc_num_outputs)
    drop2 = tf.nn.dropout(fc2, keep_prob)
    fc3 = fully_conn(drop2, fc_num_outputs)

    # Output Layer
    out = output(fc3, 10)

    # return output that represents logits
    return out


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # run tensor flow session using train_neural_network parameters
    session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # calculate loss & validation
    loss = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
    valid_acc = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})

    # print loss & validation
    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))


def train_model():
    print('Training...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())

        # Training cycle
        for epoch in range(epochs):
            # Loop over all batches
            n_batches = 5
            for batch_i in range(1, n_batches + 1):
                for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                    train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                print_stats(sess, batch_features, batch_labels, cost, accuracy)

        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)



def test_model():
    """
    Test the saved model against the test dataset
    """

    # Set batch size if not already set
    try:
        if batch_size:
            pass
    except NameError:
        batch_size = 64


    n_samples = 4
    top_n_predictions = 3

    test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0

        for test_feature_batch, test_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


if __name__ == '__main__':

    # Preprocess Training, Validation, and Testing Data
    helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

    # Load the Preprocessed Validation data
    valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

    # Hyperparameters
    epochs = 25
    batch_size = 256
    keep_probability = 0.75

    ##############################
    ## Build the Neural Network ##
    ##############################

    # Remove previous weights, bias, inputs, etc..
    tf.reset_default_graph()

    # Inputs
    x = neural_net_image_input((32, 32, 3))
    y = neural_net_label_input(10)
    keep_prob = neural_net_keep_prob_input()

    # Model
    logits = conv_net(x, keep_prob)

    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    # train model on graph
    train_model()

    # test model on saved graph
    test_model()
