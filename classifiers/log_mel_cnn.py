import json
from statistics import mean
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import random

DATA_PATH = 'C:\\Users\\semih\\OneDrive\\Uni\\Project\\Code\\data\\log_mel_data.json'


def load_data(data_path):
    with open(data_path, 'r') as fp:
        data = json.load(fp)

    #X = np.array(data['mfcc'],dtype=object)
    X = np.array(data['mel'])
    Y = np.array(data['labels'])
    return X, Y


def prepare_datasets(test_size, validation_size):
    # load in the data
    X, Y = load_data(DATA_PATH)

    # # create train/test split
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=test_size)

    healthy = []
    pathological = []

    for i in range(len(Y)):
        if Y[i]:
            pathological.append(i)
        else:
            healthy.append(i)
    # Indices of random group of 83 samples from path and healthy
    healthy_test = random.sample(healthy, 83)
    path_test = random.sample(pathological, 83)

    all_test = healthy_test + path_test

    # Remaining indices that are not in test set
    healthy_train = [x for x in healthy if x not in healthy_test]
    path_train = [x for x in pathological if x not in path_test]

    all_train = healthy_train + path_train

    X_test = []
    X_train = []
    Y_test = []
    Y_train = []

    for index in all_test:
        X_test.append(X[index])
        Y_test.append(Y[index])

    for index in all_train:
        X_train.append(X[index])
        Y_train.append(Y[index])

    X_test, X_train, Y_test, Y_train = np.array(X_test), np.array(
        X_train), np.array(Y_test), np.array(Y_train)
    # create the train/validation split
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train, test_size=validation_size)

    # 3d array for each sample needed in tensorflow
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test


def build_model(input_shape):

    # create model
    # CNN with 3 convolutional layers followed by max pooling
    model = keras.Sequential()

    # 1st conv layer - relu is recitified linear unit -- research
    model.add(keras.layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=input_shape))
    # max pooling layer
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=input_shape))
    # max pooling layer
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer - kernel size and max pooling size changed
    model.add(keras.layers.Conv2D(
        32, (2, 2), activation='relu', input_shape=input_shape))
    # max pooling layer
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed to dense layer
    model.add(keras.layers.Flatten())
    # add dense layer - fully connected layer for classification - 64 neurons used
    model.add(keras.layers.Dense(64, activation='relu'))
    # dropout for combatting overfitting
    model.add(keras.layers.Dropout(0.3))

    # output layer - number of neurons here equals number of classification groups
    model.add(keras.layers.Dense(2, activation='softmax'))

    return model


def predict(model, X, Y):

    real_indexes = []
    predicted_indexes = []
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(X_test)):
        X = X_test[i]
        real_index = Y_test[i]

        X = X[np.newaxis, ...]
        prediction = model.predict(X)  # needs 4D array
        # 1D array which tells us the prediction
        predicted_index = np.argmax(prediction, axis=1)
        real_indexes.append(real_index)
        predicted_indexes.append(predicted_index)

    for j in range(len(real_indexes)):
        real = int(real_indexes[j])
        predicted = int(predicted_indexes[j][0])

        # if expected 1 and predicted 0
        if real == 1 and predicted != 1:
            false_negatives += 1
        # if expected 1 and predicted 1
        elif real == 1 and predicted == 1:
            true_positives += 1
        # if expected 0 and predicted 1
        elif real != 1 and predicted == 1:
            false_positives += 1
        # if expected 0 and predicted 0
        else:
            true_negatives += 1
        #print(type(real), type(predicted))
        #print(real, predicted)

    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    return sensitivity, specificity


if __name__ == '__main__':

    accuracy_list = []
    specificity_list = []
    sensitivity_list = []
    for test_number in range(10):
        # create train, validation and test sets
        X_train, X_validation, X_test, Y_train, Y_validation, Y_test = prepare_datasets(
            0.25, 0.2)

        # build the CNN net
        print(X_train.shape)
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

        print('Input shape for model is: ', input_shape)
        model = build_model(input_shape)

        # compile the network
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # train the CNN
        model.fit(X_train, Y_train, validation_data=(
            X_validation, Y_validation), batch_size=32, epochs=30)

        # evaluate the CNN on test set
        test_error, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
        print('Accuracy on test set is: {}'.format(test_accuracy))

        sensitivity, specificity = predict(model, X_test, Y_test)
        print("Sensitivity = {}".format(sensitivity))
        print("Specificity = {}".format(specificity))
        accuracy_list.append(test_accuracy)
        specificity_list.append(specificity)
        sensitivity_list.append(sensitivity)

    mean_accuracy = mean(accuracy_list)
    mean_sensitivity = mean(sensitivity_list)
    mean_specificity = mean(specificity_list)
    print("Mean Accuracy = {}".format(mean_accuracy))
    print("Mean Sensitivity = {}".format(mean_sensitivity))
    print("Mean Specificity = {}".format(mean_specificity))
