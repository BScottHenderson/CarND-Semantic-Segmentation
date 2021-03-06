#!/usr/bin/env python3
import os
import time
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import numpy as np
import scipy
import cv2
from moviepy.editor import VideoFileClip

# Option Flags
PROCESS_TEST_IMAGES = False
PROCESS_VIDEO_FILE  = True
WRITE_OUTPUT_FRAMES = False

# Model parameters
REGULARIZER_SCALE = 1e-3

# Training parameters
EPOCHS        = 50
BATCH_SIZE    = 8       # Keep batch size low to avoid OOM (out-of-memory) errors.
KEEP_PROB     = 0.5     # Always use 1.0 for validation, this is for training.
LEARNING_RATE = 0.00075

# Image text parameters
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_COLOR = (255, 255, 255)
TEXT_THICKNESS = 2
TEXT_LINE_TYPE = cv2.LINE_AA


# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))
print('Keras Version     : {}'.format(tf.keras.__version__))

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
print('Test load_vgg().')
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
    # num_classes = 2

    """
    Note: use 'kernel_regularizer' for all Conv2D[Transpose] layers below to keep weights
    from becoming too large. This helps to mitigate the tendency to run out of memory during
    training.
    """


    #
    # FCN-8 Encoder
    #

    # Replace the fully-connected VGG16 output layer with a 1x1 convolutional layer.
    # Assume that vgg_layer7_out is the VGG16 layer immediately preceding the final dense layer.

    # For a 1x1 convolution we use a kernel size of 1.
    output = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1, strides=(1, 1),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))(vgg_layer7_out)

    # At this point we have:
    # 1. Downsampled the input image and extracted features using the VGG16 encoder.
    # 2. Replaced the linear/fully-connected/dense layer with a 1x1 convolution.

    #
    # FCN-8 Decoder
    #

    # First upsample the image back to the original image size (deconvolution).
    output = tf.keras.layers.Conv2DTranspose(filters=num_classes, kernel_size=4, strides=(2, 2), padding='SAME',
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))(output)

    # Add a Print() node to the graph. Use tf.shape() instead of output.get_shape() so that
    # the shape will be the runtime shape not the static shape (which has not yet been set).
    # tf.Print(output, [tf.shape(output)])

    # Add a skip connection to vgg_layer4_out.
    vgg_layer4_out_1x1 = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1, strides=(1, 1),
                                                kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))(vgg_layer4_out)
    output = tf.add(output, vgg_layer4_out_1x1)
    output = tf.keras.layers.Conv2DTranspose(filters=num_classes, kernel_size=4, strides=(2, 2), padding='SAME',
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))(output)

    vgg_layer3_out_1x1 = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1, strides=(1, 1),
                                                kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))(vgg_layer3_out)
    output = tf.add(output, vgg_layer3_out_1x1)
    output = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=16, strides=(8, 8), padding='SAME',
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))(output)

    # output layer shape should be [None, None, None, num_classes]
    return output
print('Test layers().')
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
    # The output tensor is 4D (BHWC) so we have to reshape it to 2D:
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # Reshape to (-1, num_classes) will fill in the first value such that the total size remains constant.
    # logits is now a 2D tensor where each row represents a pixel and each column represents a class

    # To apply standard cross-entropy loss:
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=correct_label))

    # Training op:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
print('Test optimize().')
tests.test_optimize(optimize)


def evaluate(X_data, y_data, x, y, accuracy_operation):
    """
    Evalutation function used for training.
    """
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset : offset + BATCH_SIZE], y_data[offset : offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


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
        total_loss = 0
        batch_num = 0
        for image, label in get_batches_fn(batch_size):
            batch_num += 1
            # test, loss = sess.run([train_op, cross_entropy_loss],
            _, loss = sess.run([train_op, cross_entropy_loss],
                     feed_dict={ input_image: image,
                                 correct_label: label,
                                 keep_prob: KEEP_PROB,
                                 learning_rate: LEARNING_RATE })
            # print('Batch {:2} loss: {}'.format(batch_num, loss))
            total_loss += loss
        epoch_loss = total_loss / batch_num
        print('Epoch {0} average batch loss: {1:.2}'.format(i+1, epoch_loss))
print('Test train_nn().')
tests.test_train_nn(train_nn)


class SemanticSegmentation():
    """
    Define a class to encapsulate video processing for semantic segmentation.
    """

    def __init__(self, sess, logits, keep_prob, image_pl, image_shape):
        # Current frame number.
        self.current_frame = 0
        # Output dir for modified video frames.
        self.video_dir = None
        # TF session
        self.sess = sess
        # TF Tensor for the logits (trained model)
        self.logits = logits
        # TF Placeholder for the dropout keep probability
        self.keep_prob = keep_prob
        # TF Placeholder for the image
        self.image_pl = image_pl
        # Shape of images expected by model.
        self.image_shape = image_shape

    def ProcessVideoClip(self, input_file, video_dir=None):
        """
        Apply the Classify() function to each frame in a given video file.
        Save the results to a new video file in the same location using the
        same filename but with "_class" appended.

        Args:
            input_file (str): Process this video file.
            video_dir (str): Optional location for modified video frames.

        Returns:
            none

        To speed up the testing process or for debugging we can use a subclip
        of the video. To do so add

            .subclip(start_second, end_second)

        to the end of the line below, where start_second and end_second are
        integer values representing the start and end of the subclip.
        """
        self.video_dir = video_dir

        # Open the video file.
        input_clip = VideoFileClip(input_file)  # .subclip(40, 45)

        # For each frame in the video clip, replace the frame image with the
        # result of applying the 'Classify' function.
        # NOTE: this function expects color images!!
        self.current_frame = 0
        output_clip = input_clip.fl(self.Classify)

        # Save the resulting, modified, video clip to a file.
        file_name, ext = os.path.splitext(input_file)
        output_file = file_name + '_classified' + ext
        output_clip.write_videofile(output_file, audio=False)

        # Cleanup
        input_clip.reader.close()
        input_clip.audio.reader.close_proc()
        del input_clip
        output_clip.reader.close()
        output_clip.audio.reader.close_proc()
        del output_clip

    def Classify(self, get_frame, t):
        """
        Given an image (video frame) run a trained semantic segmentation model.
        Draw the results on a copy of the input image and return the result.

        Args:
            get_frame: Video clip's get_frame method.
            t: time in seconds

        Returns:
            copy of the input image with classes drawn
        """
        self.current_frame += 1

        # Get the current video frame as an image and resize to the image shape
        # expected by the model.
        image = scipy.misc.imresize(get_frame(t), self.image_shape)

        # Run inference
        im_softmax = self.sess.run(
            [tf.nn.softmax(self.logits)],
            {self.keep_prob: 1.0, self.image_pl: [image]})
        # Splice out second column (road), reshape output back to image_shape
        im_softmax = im_softmax[0][:, 1].reshape(self.image_shape[0], self.image_shape[1])
        # If road softmax > 0.5, prediction is road
        segmentation = (im_softmax > 0.5).reshape(self.image_shape[0], self.image_shape[1], 1)
        # Create mask based on segmentation to apply to original image
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        # Convert to numpy array.
        image = np.array(street_im)

        # Write the frame number to the image.
        frame = 'Frame: {}'.format(self.current_frame)
        cv2.putText(image, frame, (1050, 30),
                    TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, TEXT_LINE_TYPE)

        # Write the time (parameter t) to the image.
        time = 'Time: {}'.format(int(round(t)))
        cv2.putText(image, time, (1050, 700),
                    TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, TEXT_LINE_TYPE)

        # Optionally write the modified image to a file.
        if self.video_dir is not None:
            output_file = os.path.join(self.video_dir,
                                       'frame{:06d}.jpg'.format(self.current_frame))
            cv2.imwrite(output_file, image)

        # Return the modified image.
        return image


def run():
    start = time.process_time()
    print('Start of run() ...')
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    model_dir = './model'

    # Disable this test since we've added augmented images to the data dir.
    # tests.test_for_kitti_dataset(data_dir)

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        print('Create the FCN-8 model ...')

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

        # TODO: Train NN using the train_nn function (part 1)
        print('Create training operation ...')
        logits, train_op, cross_entropy_loss = optimize(fcn8_output, correct_label, learning_rate, num_classes)

        saver = tf.train.Saver()
        checkpoint_file_exists = os.path.isfile(os.path.join(model_dir, 'checkpoint'))
        if checkpoint_file_exists:
            print('Restore trained model from checkpoint ...')
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        else:
            # TODO: Train NN using the train_nn function (part 2)
            print('Train FCN-8 model ...')
            train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, vgg_input, correct_label, vgg_keep_prob, learning_rate)

            print('Save the trained FCN-8 model ...')
            saver.save(sess, os.path.join(model_dir, 'fcn8'))
            print('Model saved.')

        proc = SemanticSegmentation(sess, logits, vgg_keep_prob, vgg_input, image_shape)

        # TODO: Save inference data using helper.save_inference_samples
        if PROCESS_TEST_IMAGES:
            print('Save sample images ...')
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        # OPTIONAL: Apply the trained model to a video
        if PROCESS_VIDEO_FILE:
            video_dir = './data/AdvancedLaneLinesFrames' if WRITE_OUTPUT_FRAMES else None
            proc.ProcessVideoClip('./data/AdvancedLaneLinesVideos/project_video.mp4', video_dir)
            # proc.ProcessVideoClip('./data/AdvancedLaneLinesVideos/challenge_video.mp4', video_dir)
            # proc.ProcessVideoClip('./data/AdvancedLaneLinesVideos/harder_challenge_video.mp4', video_dir)

    secs = time.process_time() - start
    mins = secs / 60
    secs = round(secs % 60, 2)
    hrs  = int(mins // 60)
    mins = int(mins % 60)
    print('Elapsed time: {}h {}m {}s'.format(hrs, mins, secs))
    print('Done.')

if __name__ == '__main__':
    run()
