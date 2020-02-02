import pickle
from keras import layers, models
from keras import optimizers
from keras import losses, metrics
from keras import callbacks
from TrackGen import FileManager
import keras.backend as K
import tensorflow as tf
import numpy as np

data_dir = "/run/media/whosuka/windows/data"


def train_model(gen_train, gen_valid, idx):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(None, 4)))
    model.add(layers.Masking(mask_value=0., input_shape=(None, 4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Bidirectional(layers.GRU(16, return_sequences=True),
                                   merge_mode='ave'))
    model.add(layers.BatchNormalization())
    model.add(layers.Bidirectional(layers.GRU(16, return_sequences=True),
                                   merge_mode='ave'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(16))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(2, activation='sigmoid'))

    model.summary()

    callbacks_list = [callbacks.EarlyStopping(monitor="val_symmetric_accuracy",
                                              patience=1,
                                              min_delta=0.001,
                                              mode='max'),
                      callbacks.ModelCheckpoint(filepath="track_%i_weights.h5" % idx,
                                                monitor="val_symmetric_accuracy",
                                                save_best_only=True,
                                                save_weights_only=True),
                      callbacks.ReduceLROnPlateau(monitor="val_symmetric_accuracy",
                                                  factor=0.5,
                                                  mode='max',
                                                  min_delta=0.001,
                                                  patience=1)]

    model.compile(optimizer=optimizers.Adam(lr=0.002),
                  loss=losses.binary_crossentropy,
                  metrics=[symmetric_accuracy, mask_accuracy])

    out = model.fit_generator(gen_train,
                              steps_per_epoch=len(gen_train),
                              epochs=50,
                              callbacks=callbacks_list,
                              validation_data=gen_valid,
                              validation_steps=len(gen_valid))

    model.save('model_tracker_%i.h5' % idx)
    with open("history_track_%i.pkl" % idx, "wb") as file:
        pickle.dump(out.history, file)


def test_model(x_test, y_test):
    model = models.load_model('model_tracker_10.h5')

    print(model.evaluate(x=x_test,
                         y=y_test,
                         batch_size=1000))


def symmetric_logloss(y_true, y_pred):
    not_y_true = tf.gather(y_true, K.constant([1, 0], dtype=np.int32), axis=-1)
    return losses.binary_crossentropy(y_true, y_pred) * losses.binary_crossentropy(not_y_true, y_pred)


def symmetric_accuracy(y_true, y_pred):
    idx = tf.where(tf.not_equal(y_true, -1.0))
    not_y_true = tf.gather(y_true, K.constant([1, 0], dtype=np.int32), axis=-1)
    y_true = tf.gather_nd(y_true, idx)
    y_pred = tf.gather_nd(y_pred, idx)
    not_y_true = tf.gather_nd(not_y_true, idx)
    return K.maximum(metrics.binary_accuracy(y_true, y_pred), metrics.binary_accuracy(not_y_true, y_pred))


def mask_accuracy(y_true, y_pred):
    idx = tf.where(tf.not_equal(y_true, -1.0))
    y_true = tf.gather_nd(y_true, idx)
    y_pred = tf.gather_nd(y_pred, idx)
    return metrics.binary_accuracy(y_true, y_pred)


if __name__ == "__main__":
    # ------------------------------
    # test model
    # x_train, y_train, x_valid, y_valid = data(train_dir + os.sep + 'random' + os.sep + 'good',
    #                                           valid_dir + os.sep + 'random' + os.sep + 'good')
    # test_model(x_train, y_train)
    # ------------------------------

    # ------------------------------
    # train model
    idx = 1
    fm = FileManager(data_dir)
    gen_train, gen_valid = fm.get_train_gen(), fm.get_valid_gen()
    train_model(gen_train, gen_valid, idx)
    # ------------------------------

    # ------------------------------
    # make predictions
    # x_train, y_train, x_valid, y_valid = data(train_dir + os.sep + 'random' + os.sep + 'good',
    #                                           valid_dir + os.sep + 'random' + os.sep + 'good', axis='z')
    # model = models.load_model('model_tracker_18.h5')
    # predict_labels = model.predict(x_valid, batch_size=1000)
    # out_dir = valid_dir + os.sep + 'random' + os.sep + 'good'
    # np.save(out_dir + os.sep + "predict_labels_z.npy", predict_labels, allow_pickle=False)
    # ------------------------------

    # ------------------------------
    # from keras.utils import plot_model
    # model = create_model()
    # plot_model(model, to_file='tracker.pdf',
    #            show_shapes=True, show_layer_names=False)
