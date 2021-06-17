from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.losses import mean_squared_logarithmic_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn import metrics
import numpy as np

import datasets
from model_interface import ModelInterface

BATCH_SIZE = 256
EPOCHS = 100
VAL_SIZE = 0.2


class HistogramDNNModel(ModelInterface):
    NORMALIZE = False
    DISTRIBUTION = 'all'
    MATCHED = True
    SCALE = 'Small'
    MINUS_ONE = False

    def train(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
              histogram_path=None) -> None:
        """
        Train a regression model for spatial join cost estimator, then save the trained model to file
        """

        # Extract train and test data, but only use train data
        # target = 'join_selectivity'
        num_rows, num_columns = 32, 32
        y, ds1_histograms, ds2_histograms, ds_bops_histogram = datasets.load_histogram_features(join_result_path, tabular_path, histogram_path, num_rows, num_columns)
        y_train, y_test, ds1_histograms_train, ds1_histograms_test, ds2_histograms_train, ds2_histograms_test, ds_bops_histogram_train, ds_bops_histogram_test \
            = train_test_split(y, ds1_histograms, ds2_histograms, ds_bops_histogram, test_size=0.2, random_state=42)

        # model = HistogramDNNModel.create_cnn(num_rows, num_columns, 1, regress=True)

        # create CNN model
        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(num_rows, num_columns, 1)))
        model.add(Conv2D(8, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='linear'))

        EPOCHS = 40
        LR = 1e-2
        opt = Adam(lr=LR, decay=LR / EPOCHS)

        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=10,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )

        model.compile(metrics=['mean_absolute_percentage_error'], loss="mean_absolute_percentage_error", optimizer=opt)
        model.fit(
            ds_bops_histogram_train, y_train,
            validation_data=(ds_bops_histogram_test, y_test),
            epochs=EPOCHS, batch_size=256, callbacks=[early_stopping])

        y_pred = model.predict(ds_bops_histogram_test)

        # Convert back to 1 - y if need
        if HistogramDNNModel.MINUS_ONE:
            y_test, y_pred = 1 - y_test, 1 - y_pred

        # Compute accuracy metrics
        mse = metrics.mean_squared_error(y_test, y_pred)
        mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
        msle = np.mean(mean_squared_logarithmic_error(y_test, y_pred))
        mae = metrics.mean_absolute_error(y_test, y_pred)
        print('mae: {}\nmape: {}\nmse: {}\nmlse: {}'.format(mae, mape, mse, msle))
        print('{}\t{}\t{}\t{}'.format(mae, mape, mse, msle))

    def test(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
             histogram_path=None) -> (float, float, float, float):
        """
        Evaluate the accuracy metrics of a trained  model for spatial join cost estimator
        :return mean_squared_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, mean_absolute_error
        """
        pass

    def create_cnn(width, height, depth, filters=(4, 8, 16), regress=False):
        # initialize the input shape and channel dimension, assuming
        # TensorFlow/channels-last ordering
        input_shape = (height, width, depth)
        chan_dim = -1

        # define the model input
        inputs = Input(shape=input_shape)

        # loop over the number of filters
        for (i, f) in enumerate(filters):
            # if this is the first CONV layer then set the input
            # appropriately
            if i == 0:
                x = inputs

            # CONV => RELU => BN => POOL
            x = Conv2D(f, (3, 3), padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chan_dim)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # flatten the volume, then FC => RELU => BN => DROPOUT
        x = Flatten()(x)
        x = Dense(8)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Dropout(0.5)(x)

        # apply another FC layer, this one to match the number of nodes
        # coming out of the MLP
        x = Dense(4)(x)
        x = Activation("relu")(x)

        # check to see if the regression node should be added
        if regress:
            x = Dense(1, activation="linear")(x)

        # construct the CNN
        model = Model(inputs, x)

        # return the CNN
        return model
