#!/usr/bin/env python3
import os
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Model parameters
REGULARIZER_SCALE = 1e-3

# Training parameters
EPOCHS        = 6
BATCH_SIZE    = 16      # Keep batch size low to avoid OOM (out-of-memory) errors.
KEEP_PROB     = 0.5     # Always use 1.0 for validation, this is for training.
LEARNING_RATE = 0.001


# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name      = 'image_input:0'
    vgg_keep_prob_tensor_name  = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    vgg_input      = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob  = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # print('vgg_layer3_out.shape: {}'.format(vgg_layer3_out.shape))  # 256
    # print('vgg_layer4_out.shape: {}'.format(vgg_layer4_out.shape))  # 512
    # print('vgg_layer7_out.shape: {}'.format(vgg_layer7_out.shape))  # 4096
    # print('num_classes = {}'.format(num_classes))   # 2

    """
    Note: use 'kernel_regularizer' for all conv2d[_transpose] layers below to keep weights
    from becoming too large. This helps to mitigate the tendency to run out of memory during
    training.
    """


    #
    # FCN-8 Encoder
    #

    # Replace the fully-connected VGG16 output layer with a 1x1 convolutional layer.
    # Assume that vgg_layer7_out is the VGG16 layer immediately preceding the final dense layer.

    # For a 1x1 convolution we use a kernel size of 1.
    output = tf.layers.conv2d(vgg_layer7_out, filters=num_classes, kernel_size=1, strides=(1, 1),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))

    # At this point we have:
    # 1. Downsampled the input image and extracted features using the VGG16 encoder.
    # 2. Replaced the linear/fully-connected/dense layer with a 1x1 convolution.

    #
    # FCN-8 Decoder
    #

    # First upsample the image back to the original image size (deconvolution).
    output = tf.layers.conv2d_transpose(output, num_classes, kernel_size=4, strides=(2, 2), padding='SAME',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))

    # Add a Print() node to the graph. Use tf.shape() instead of output.get_shape() so that
    # the shape will be the runtime shape not the static shape (which has not yet been set).
    tf.Print(output, [tf.shape(output)])

    # Add a skip connection to vgg_layer4_out.
    vgg_layer4_out_1x1 = tf.layers.conv2d(vgg_layer4_out, filters=num_classes, kernel_size=1, strides=(1, 1),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))
    output = tf.add(output, vgg_layer4_out_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, kernel_size=4, strides=(2, 2), padding='SAME',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))

    vgg_layer3_out_1x1 = tf.layers.conv2d(vgg_layer3_out, filters=num_classes, kernel_size=1, strides=(1, 1),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))
    output = tf.add(output, vgg_layer3_out_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, kernel_size=16, strides=(8, 8), padding='SAME',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))

    # output layer shape should be [None, None, None, num_classes]
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    # The output tensor is 4D so we have to reshape it to 2D:
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # logits is now a 2D tensor where each row represents a pixel and each column represents a class

    # To apply standard cross-entropy loss:
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    # Training op:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        print('Epoch: {}'.format(i + 1))
        batch_num = 0
        for image, label in get_batches_fn(batch_size):
            batch_num += 1
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={ input_image: image,
                                           correct_label: label,
                                           keep_prob: KEEP_PROB,
                                           learning_rate: LEARNING_RATE })
            print('Batch {:2} loss: {}'.format(batch_num, loss))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # TODO: Build NN using load_vgg, layers, and optimize function
        print('Load VGG ...')
        vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        print('Define the FCN-8 model ...')
        fcn8_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        # TODO: Train NN using the train_nn function
        print('Create training operation ...')
        logits, train_op, cross_entropy_loss = optimize(fcn8_output, correct_label, learning_rate, num_classes)
        print('Train ...')
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, vgg_input, correct_label, vgg_keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        print('Save sample images ...')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        # OPTIONAL: Apply the trained model to a video

        """
        Note: The training data includes three classes, not just two.
        1. Road in front of the vehicle.
        2. Not a road
        3. Road that is not in front of the vehicle.

        Modify helper.gen_batch_function() to create a label for all three classes.
        The initial implementation just separates the image into red and not-red.
        Modify to look for black as well (or pink, but black is easier to represent as rgb).

        Existing implementaion labels red pixesl as 'false', everything else is 'true'.
        Modify to use an integer class, e.g., red==0, black==1, pink/other=2
        """


if __name__ == '__main__':
    run()
