# import keras
import tensorflow as tf
import cv2
import numpy as np
import sys
import os
import argparse
import scipy.io as sio

default_model_name = 'keras_alexnet.h5'
default_model_dir = 'models'

keraslayers = tf.keras.layers

initializer = tf.keras.initializers.random_normal(0.0, 0.01)
regularizer = tf.keras.regularizers.l2(5e-4)

def build_model(class_count=10):
    model = tf.keras.Sequential(
        [
            keraslayers.Conv2D(
                64, 11, 4,
                padding='same',
                activation=tf.keras.activations.relu,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                input_shape=(32,32,3),
                data_format='channels_last'
            ),
            keraslayers.MaxPooling2D(
                2, 2, padding='valid'
            ),
            keraslayers.Conv2D(
                192, 5,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.keras.activations.relu
            ),
            keraslayers.MaxPooling2D(
                2, 2, padding='valid'
            ),
            keraslayers.Conv2D(
                384, 3,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.keras.activations.relu
            ),
            keraslayers.Conv2D(
                256, 3,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.keras.activations.relu
            ),
            keraslayers.Conv2D(
                256, 3,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.keras.activations.relu
            ),
            keraslayers.MaxPooling2D(
                2, 2, padding='valid'
            ),
            keraslayers.Flatten(),
            keraslayers.Dropout(0.3),
            keraslayers.Dense(
                class_count,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.softmax
            )
        ]
    )
    return model


def preprocess_image(image, image_height=224, image_width=224):
    """resize images to the appropriate dimensions
    :param image_width:
    :param image_height:
    :param image: image
    :return: image
    """
    return cv2.resize(image, (image_height, image_width))


def load_dataset():
    """loads training and testing resources
    :return: x_train, y_train, x_test, y_test
    """
    
    def normalize(f, means, stddevs):
        """
        Normalizes data using means and stddevs
        """
        normalized = (f/255 - means) / stddevs
        return normalized

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data() 

    x_target_train = X_train[16000:32000]
    y_target_train = y_train[16000:32000]
    x_target_test = X_test[4000:8000]
    y_target_test = y_test[4000:8000]

    x_target_train = normalize(x_target_train, [
        0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    x_target_test = normalize(x_target_test, [
        0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])



    return (x_target_train,y_target_train),(x_target_test,y_target_test)


def generator(batch_size, class_count, x_data, y_data):
    """generates batch training (and evaluating) data and labels
    """
    while True:
        X = []  # batch training set
        Y = []  # batch labels
        for index in range(0, len(x_data)):
            # X.append(preprocess_image(x_data[index], image_height, image_width))
            X.append(x_data[index])
            Y.append(y_data[index])
            if (index + 1) % batch_size == 0:
                yield np.array(X), tf.keras.utils.to_categorical(np.array(Y), class_count)
                X = []
                Y = []


def train_model(model, class_count=10, epochs=90):
    """train the SuperVision/alexnet NN model
    :param epochs:
    :param image_height:
    :param class_count:
    :param image_width:
    :param model: NN model (uncompiled, without weights)
    :return: compiled NN model with weights
    """
    # compile with SGD optimizer and categorical_crossentropy as the loss function
    # model.compile(loss="categorical_crossentropy",
    #               optimizer=tf.keras.optimizers.SGD(lr=0.001),
    #               metrics=['accuracy'])
    model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

    # training parameters
    (x_train, y_train), (x_test, y_test) = load_dataset()
    batch_size = 64
    steps = len(x_train) / batch_size

    # train the model using a batch generator
    import time
    tic = time.time()
    batch_generator = generator(batch_size, class_count, x_train, y_train)
    model.fit_generator(generator=batch_generator,
                        steps_per_epoch=steps,
                        epochs=epochs,
                        verbose=1)
    toc = time.time()
    print("cost time:%ss"%(str(toc - tic)))
    # train the model on the dataset
    # count=10000
    # x_train = np.array([preprocess_image(image) for image in x_train[:count]])
    # y_train = tf.keras.utils.to_categorical(y_train[:count], class_count)
    # model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)


def evaluate(model, class_count=100):
    """evaluate the performance of the trained model using the prepared testing set
    :param image_width:
    :param class_count:
    :param image_height:
    :param model: compiled NN model with trained weights
    """

    # training parameters
    (x_train, y_train), (x_test, y_test) = load_dataset()
    batch_size = 128
    steps = len(x_test) / batch_size

    # train the model using a batch generator
    batch_generator = generator(batch_size, class_count, x_test, y_test)
    scores = model.evaluate_generator(generator=batch_generator,
                                      #verbose=1,
                                      steps=steps)
    print(scores)
    print("Test Loss:\t", scores[0])
    print("Test Accuracy:\t", scores[1])


def parse_arguments():
    """parse command line input
    :return: dictionary of arguments keywords and values
    """
    parser = argparse.ArgumentParser(description="Construct and train an alexnet model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n',
                        default=default_model_name,
                        metavar='<model_name>',
                        help='The name to be given to the output model.')
    parser.add_argument('-d',
                        default=default_model_dir,
                        metavar='<output_directory>',
                        help='The directory in which the models should be saved.')
    parser.add_argument('-e',
                        default=100,
                        metavar='<number_of_epochs>',
                        help='The number of epochs used to train the model. The original alexnet used 90 epochs.')
    return vars(parser.parse_args())


def main():
    """build, train, and test an implementation of the alexnet CNN model in tf.keras.
    This model is trained and tested on the CIFAR-100 dataset
    """
    # parse arguments
    args = parse_arguments()
    save_dir = os.path.join(os.getcwd(), args['d'])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_name = "keras_alexnet_shadow_model.h5"
    model_path = os.path.join(save_dir, model_name)
    epochs = 200

    # build and train the model
    model = build_model(class_count=100)
    print(model.summary())
    train_model(model, class_count=100, epochs=epochs)

    # test the model
    evaluate(model, class_count=100)

    # save the trained model
    model.save(model_path)
    print("Alexnet model saved to: %s" % model_path)


def get_outputs():
    model = tf.keras.models.load_model("models/keras_alexnet_shadow_model.h5")
    (x_train, y_train), (x_test, y_test) = load_dataset()
    # batch_generator = generator(64, 10, 224, 224, x_test, y_test)
    batch_size = 1
    class_count = 100
    steps = len(x_train) / batch_size
    member_batch_generator = generator(batch_size, class_count, x_train, y_train)
    score_member = model.predict_generator(generator=member_batch_generator,
                                      #verbose=1,
                                      steps=steps)

    steps = len(x_test) / batch_size
    non_member_batch_generator = generator(batch_size, class_count, x_test, y_test)
    score_non_member = model.predict_generator(generator=non_member_batch_generator,
                                      #verbose=1,
                                      steps=steps)
    print(score_member.shape)
    print(score_non_member.shape)

    import pickle
    pickle.dump((score_member,y_train),open("score_member_shadow.pkl","wb"))
    pickle.dump((score_non_member,y_test),open("score_non_member_shadow.pkl","wb"))
    # pickle.dump((score_non_member,y_test),open("score_non_member_user.pkl","wb"))


if __name__ == "__main__":
    # execute only if run as a script
    try:
        # sys.exit(main())
        get_outputs()
    except KeyboardInterrupt:
        sys.exit(1)