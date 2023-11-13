import keras
import cv2
import numpy as np
import sys
import os
import argparse
import scipy.io as sio

default_model_name = 'keras_alexnet_shadow_model.h5'
default_model_dir = 'models'


def build_model(image_height=224, image_width=224, class_count=10):
    model = keras.models.Sequential()

    # layer 1 - "filters the 224 x 224 x 3 input image with 96 kernels
    #           of size 11 x 11 x 3 with a stride of 4 pixels"
    model.add(keras.layers.Conv2D(filters=96,
                                  kernel_size=(11, 11),
                                  strides=4,
                                  input_shape=(image_height, image_width, 3),
                                  activation="relu",
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3),
                                     strides=(2, 2)))

    # layer 2 - "256 kernels of size 5 x 5 x 48"
    model.add(keras.layers.Conv2D(filters=256,
                                  kernel_size=(5, 5),
                                  activation="relu",
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3),
                                     strides=(2, 2)))

    # layer 3 - "384 kernels of size 3 x 3 x 256"
    model.add(keras.layers.Conv2D(filters=384,
                                  kernel_size=(3, 3),
                                  activation="relu",
                                  padding="same"))
    # layer 4 - "384 kernels of size 3 x 3 x 192"
    model.add(keras.layers.Conv2D(filters=384,
                                  kernel_size=(3, 3),
                                  activation="relu",
                                  padding="same"))
    # layer 5 - "256 kernels of size 3 x 3 x 192"
    model.add(keras.layers.Conv2D(filters=256,
                                  kernel_size=(3, 3),
                                  activation="relu",
                                  padding="same"))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3),
                                     strides=(2, 2)))

    # flatten before feeding into FC layers
    model.add(keras.layers.Flatten())

    # fully connected layers
    # "The fully-connected layers have 4096 neurons each."
    # "We use dropout in the first two fully-connected layers..."
    model.add(keras.layers.Dense(units=4096))  # layer 6
    model.add(keras.layers.Dense(units=4096))  # layer 7
    model.add(keras.layers.Dense(units=class_count))  # layer 8

    # output layer is softmax
    model.add(keras.layers.Activation('softmax'))
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
    # return keras.datasets.cifar100.load_data(label_mode='fine')

    path = '/home/user01/exps/DAMIA/Third_stage/SVHN/dataset/svhn-data'
    train_raw = sio.loadmat(path+'/train_32x32.mat')
    test_raw = sio.loadmat(path+'/test_32x32.mat')

    # train_images = np.array(train_raw['X'][:2000])
    # train_labels = train_raw['y'][:2000]

    # test_images = np.array(test_raw['X'][:500])
    # test_labels = test_raw['y'][:500]

    # x_target_train = train_images
    # y_target_train = train_labels.astype('int64')
    # x_target_train = np.moveaxis(x_target_train, -1, 0)


    # x_target_test = test_images
    # y_target_test = test_labels.astype('int64')
    # x_target_test = np.moveaxis(x_target_test, -1, 0)

    
    # x_target_train = x_target_train.astype('float64')
    # x_target_test = x_target_test.astype('float64')

    # x_target_train /= 255.0
    # x_target_test /= 255.0


    train_images = np.array(train_raw['X'])
    train_labels = train_raw['y']
    test_images = np.array(test_raw['X'])
    test_labels = test_raw['y']

    train_images = np.transpose(train_raw["X"], (3, 0, 1, 2))
    test_images = np.transpose(test_raw["X"], (3, 0, 1, 2))

    train_labels = train_raw["y"]
    train_labels[train_labels == 10] = 0
    test_labels = test_raw["y"]
    test_labels[test_labels == 10] = 0

    scalar = 1 / 255.
    train_images = train_images * scalar
    test_images = test_images * scalar

    x_target_train = train_images[20000:40000].astype('float64')
    y_target_train = train_labels[20000:40000].flatten()
    x_target_test = train_images[5000:10000].astype('float64')
    y_target_test = train_labels[5000:10000].flatten()


    return (x_target_train,y_target_train),(x_target_test,y_target_test)


def generator(batch_size, class_count, image_height, image_width, x_data, y_data):
    """generates batch training (and evaluating) data and labels
    """
    while True:
        X = []  # batch training set
        Y = []  # batch labels
        for index in range(0, len(x_data)):
            X.append(preprocess_image(x_data[index], image_height, image_width))
            Y.append(y_data[index])
            if (index + 1) % batch_size == 0:
                yield np.array(X), keras.utils.to_categorical(np.array(Y), class_count)
                X = []
                Y = []


def train_model(model, image_height=224, image_width=224, class_count=10, epochs=90):
    """train the SuperVision/alexnet NN model
    :param epochs:
    :param image_height:
    :param class_count:
    :param image_width:
    :param model: NN model (uncompiled, without weights)
    :return: compiled NN model with weights
    """
    # compile with SGD optimizer and categorical_crossentropy as the loss function
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=0.001),
                  metrics=['accuracy'])

    # training parameters
    (x_train, y_train), (x_test, y_test) = load_dataset()
    batch_size = 64
    steps = len(x_train) / batch_size

    # train the model using a batch generator
    import time
    tic = time.time()
    batch_generator = generator(batch_size, class_count, image_height, image_width, x_train, y_train)
    model.fit_generator(generator=batch_generator,
                        steps_per_epoch=steps,
                        epochs=epochs,
                        verbose=1)
    toc = time.time()
    print("cost time:%ss"%(str(toc - tic)))
    # train the model on the dataset
    # count=10000
    # x_train = np.array([preprocess_image(image) for image in x_train[:count]])
    # y_train = keras.utils.to_categorical(y_train[:count], class_count)
    # model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)


def evaluate(model, class_count=1000, image_height=224, image_width=224):
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
    batch_generator = generator(batch_size, class_count, image_height, image_width, x_test, y_test)
    scores = model.evaluate_generator(generator=batch_generator,
                                      #verbose=1,
                                      steps=steps)
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
                        default=15,
                        metavar='<number_of_epochs>',
                        help='The number of epochs used to train the model. The original alexnet used 90 epochs.')
    return vars(parser.parse_args())




def main():
    """build, train, and test an implementation of the alexnet CNN model in keras.
    This model is trained and tested on the CIFAR-100 dataset
    """
    # parse arguments
    args = parse_arguments()
    save_dir = os.path.join(os.getcwd(), args['d'])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, args['n'])
    epochs = int(args['e'])

    # build and train the model
    model = build_model(class_count=10)
    print(model.summary())
    train_model(model, class_count=10, epochs=epochs)

    # test the model
    evaluate(model, class_count=10)

    # save the trained model
    model.save(model_path)
    print("Alexnet model saved to: %s" % model_path)


def get_outputs():
    model = keras.models.load_model("models/keras_alexnet_shadow_model.h5")
    (x_train, y_train), (x_test, y_test) = load_dataset()
    # batch_generator = generator(64, 10, 224, 224, x_test, y_test)
    batch_size = 1
    class_count = 10
    image_height = 224
    image_width = 224
    steps = len(x_train) / batch_size
    member_batch_generator = generator(batch_size, class_count, image_height, image_width, x_train, y_train)
    score_member = model.predict_generator(generator=member_batch_generator,
                                      #verbose=1,
                                      steps=steps)

    steps = len(x_test) / batch_size
    non_member_batch_generator = generator(batch_size, class_count, image_height, image_width, x_test, y_test)
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