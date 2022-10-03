import numpy as np
from edfreader import EDFreader
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import csv
import os
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import sklearn as sk
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

seg_size = 15
n_chs = 5
drop_rate = 0.3
fs = 256

input_shape = [n_chs, int(seg_size * fs / 10), 1]



data_files = [["chb01/chb01_03", 2996, 3036],
              ["chb01/chb01_04", 1467, 1494],
              ["chb01/chb01_15", 1732, 1772],
              ["chb01/chb01_16", 1015, 1066],
              ["chb01/chb01_18", 1720, 1810],
              ["chb01/chb01_21", 327, 420],
              ["chb01/chb01_26", 1862, 1963],
              ["chb02/chb02_16", 130, 212],
              ["chb02/chb02_16+", 2972, 3053],
              ["chb02/chb02_19", 3369, 3378],
              ["chb03/chb03_01", 362, 414],
              ["chb03/chb03_02", 731, 796],
              ["chb03/chb03_03", 432, 501],
              ["chb03/chb03_04", 2162, 2214],
              ["chb03/chb03_34", 1982, 2029],
              ["chb03/chb03_35", 2592, 2656],
              ["chb03/chb03_36", 1725, 1778],
              ["chb04/chb04_05", 7804, 7853],
              ["chb04/chb04_08", 6446, 6557],
              ["chb04/chb04_28", 1679, 1781],
              ["chb04/chb04_28", 3782, 3898],  # double sez in file
              ["chb05/chb05_06", 417, 532],
              ["chb05/chb05_13", 1086, 1196],
              ["chb05/chb05_16", 2317, 2413],
              ["chb05/chb05_17", 2451, 2571],
              ["chb05/chb05_22", 2348, 2465],
              ["chb06/chb06_01", 1724, 1738],
              ["chb06/chb06_01", 7461, 7476],
              ["chb06/chb06_01", 13525, 13540],
              ["chb06/chb06_04", 327, 347],
              ["chb06/chb06_04", 6211, 6231],
              ["chb06/chb06_09", 12500, 12516],
              ["chb06/chb06_10", 10833, 10845],
              ["chb06/chb06_13", 506, 519],
              ["chb06/chb06_18", 7799, 7811],
              ["chb06/chb06_24", 9387, 9403],
              ["chb07/chb07_12", 4920, 5006],
              ["chb07/chb07_13", 3285, 3381],
              ["chb07/chb07_19", 13688, 13831],
              ["chb08/chb08_02", 2670, 2841],
              ["chb08/chb08_05", 2856, 3046],
              ["chb08/chb08_11", 2988, 3122],
              ["chb08/chb08_13", 2417, 2577],
              ["chb08/chb08_21", 2083, 2347],
              ["chb09/chb09_06", 12231, 12295],
              ["chb09/chb09_08", 2951, 3030],
              ["chb09/chb09_08", 9196, 9267],
              ["chb09/chb09_19", 5299, 5361],
              ["chb10/chb10_12", 6313, 6348],
              ["chb10/chb10_20", 6888, 6958],
              ["chb10/chb10_27", 2382, 2447],
              ["chb10/chb10_30", 3021, 3079],
              ["chb10/chb10_31", 3801, 3877],
              ["chb10/chb10_38", 4618, 4707],
              ["chb10/chb10_89", 1383, 1437],
              ["chb18/chb18_29", 3477, 3527],
              ["chb18/chb18_30", 541, 571],
              ["chb18/chb18_31", 2087, 2155],
              ["chb18/chb18_32", 1908, 1963],
              ["chb18/chb18_35", 2196, 2264],
              ["chb18/chb18_36", 463, 509],
              ["chb20/chb20_12", 94, 123],
              ["chb20/chb20_13", 1440, 1470],
              ["chb20/chb20_13", 2498, 2537],
              ["chb20/chb20_14", 1971, 2009],
              ["chb20/chb20_15", 390, 425],
              ["chb20/chb20_15", 1689, 1738],
              ["chb20/chb20_16", 2226, 2261],
              ["chb20/chb20_68", 1393, 1432],
              ["chb22/chb22_20", 3367, 3425],
              ["chb22/chb22_25", 3139, 3213],
              ["chb22/chb22_38", 1263, 1335],
              ["chb23/chb23_06", 3962, 4075],
              ["chb23/chb23_08", 325, 345],
              ["chb23/chb23_08", 5104, 5151],
              ["chb23/chb23_09", 2589, 2660],
              ["chb23/chb23_09", 6885, 6947],
              ["chb23/chb23_09", 8505, 8532],
              ["chb23/chb23_09", 9580, 9664],
              ["chb24/chb24_01", 480, 505],
              ["chb24/chb24_01", 2451, 2476],
              ["chb24/chb24_03", 231, 260],
              ["chb24/chb24_03", 2883, 2908],
              ["chb24/chb24_04", 1088, 1120],
              ["chb24/chb24_04", 1411, 1438],
              ["chb24/chb24_04", 1745, 1764],
              ["chb24/chb24_06", 1229, 1253],
              ["chb24/chb24_07", 38, 60],
              ["chb24/chb24_09", 1745, 1764],
              ["chb24/chb24_11", 3527, 3597],
              ["chb24/chb24_13", 3288, 3304],
              ["chb24/chb24_14", 1939, 1966],
              ["chb24/chb24_15", 3552, 3569],
              ["chb24/chb24_17", 3515, 3581],
              ["chb24/chb24_21", 2804, 2872],
              ]
data_files = np.array(data_files)

data_files2 = [["chb01/chb01_05", 2996, 3036],
              ["chb01/chb01_06", 1467, 1494],
              ["chb01/chb01_07", 1732, 1772],
              ["chb01/chb01_30", 1015, 1066],
              ["chb01/chb01_31", 1720, 1810],
              ["chb01/chb01_32", 327, 420],
              ["chb01/chb01_33", 1862, 1963],
              ["chb02/chb02_16", 130, 212],
              ["chb02/chb02_16+", 2972, 3053],
              ["chb02/chb02_19", 3369, 3378],
              ["chb03/chb03_01", 362, 414],
              ["chb03/chb03_02", 731, 796],
              ["chb03/chb03_03", 432, 501],
              ["chb03/chb03_04", 2162, 2214],
              ["chb03/chb03_34", 1982, 2029],
              ["chb03/chb03_35", 2592, 2656],
              ["chb03/chb03_36", 1725, 1778],
              ["chb04/chb04_05", 7804, 7853],
              ["chb04/chb04_08", 6446, 6557],
              ["chb04/chb04_28", 1679, 1781],
              ["chb04/chb04_28", 3782, 3898],  # double sez in file
              ["chb05/chb05_06", 417, 532],
              ["chb05/chb05_13", 1086, 1196],
              ["chb05/chb05_16", 2317, 2413],
              ["chb05/chb05_17", 2451, 2571],
              ["chb05/chb05_22", 2348, 2465],
              ["chb06/chb06_01", 1724, 1738],
              ["chb06/chb06_01", 7461, 7476],
              ["chb06/chb06_01", 13525, 13540],
              ["chb06/chb06_04", 327, 347],
              ["chb06/chb06_04", 6211, 6231],
              ["chb06/chb06_09", 12500, 12516],
              ["chb06/chb06_10", 10833, 10845],
              ["chb06/chb06_13", 506, 519],
              ["chb06/chb06_18", 7799, 7811],
              ["chb06/chb06_24", 9387, 9403],
              ["chb07/chb07_12", 4920, 5006],
              ["chb07/chb07_13", 3285, 3381],
              ["chb07/chb07_19", 13688, 13831],
              ["chb08/chb08_12", 2670, 2841],
              ["chb08/chb08_03", 2856, 3046],
              ["chb08/chb08_15", 2988, 3122],
              ["chb08/chb08_19", 2417, 2577],
              ["chb08/chb08_16", 2083, 2347],
              ["chb09/chb09_06", 12231, 12295],
              ["chb09/chb09_08", 2951, 3030],
              ["chb09/chb09_08", 9196, 9267],
              ["chb09/chb09_19", 5299, 5361],
              ["chb10/chb10_12", 6313, 6348],
              ["chb10/chb10_20", 6888, 6958],
              ["chb10/chb10_27", 2382, 2447],
              ["chb10/chb10_30", 3021, 3079],
              ["chb10/chb10_31", 3801, 3877],
              ["chb10/chb10_38", 4618, 4707],
              ["chb10/chb10_89", 1383, 1437],
              ["chb18/chb18_29", 3477, 3527],
              ["chb18/chb18_30", 541, 571],
              ["chb18/chb18_31", 2087, 2155],
              ["chb18/chb18_32", 1908, 1963],
              ["chb18/chb18_35", 2196, 2264],
              ["chb18/chb18_36", 463, 509],
              ["chb20/chb20_12", 94, 123],
              ["chb20/chb20_13", 1440, 1470],
              ["chb20/chb20_13", 2498, 2537],
              ["chb20/chb20_14", 1971, 2009],
              ["chb20/chb20_15", 390, 425],
              ["chb20/chb20_15", 1689, 1738],
              ["chb20/chb20_16", 2226, 2261],
              ["chb20/chb20_68", 1393, 1432],
              ["chb22/chb22_20", 3367, 3425],
              ["chb22/chb22_25", 3139, 3213],
              ["chb22/chb22_38", 1263, 1335],
              ["chb23/chb23_06", 3962, 4075],
              ["chb23/chb23_08", 325, 345],
              ["chb23/chb23_08", 5104, 5151],
              ["chb23/chb23_09", 2589, 2660],
              ["chb23/chb23_09", 6885, 6947],
              ["chb23/chb23_09", 8505, 8532],
              ["chb23/chb23_09", 9580, 9664],
              ["chb24/chb24_01", 480, 505],
              ["chb24/chb24_01", 2451, 2476],
              ["chb24/chb24_03", 231, 260],
              ["chb24/chb24_03", 2883, 2908],
              ["chb24/chb24_04", 1088, 1120],
              ["chb24/chb24_04", 1411, 1438],
              ["chb24/chb24_04", 1745, 1764],
              ["chb24/chb24_06", 1229, 1253],
              ["chb24/chb24_07", 38, 60],
              ["chb24/chb24_09", 1745, 1764],
              ["chb24/chb24_11", 3527, 3597],
              ["chb24/chb24_13", 3288, 3304],
              ["chb24/chb24_14", 1939, 1966],
              ["chb24/chb24_15", 3552, 3569],
              ["chb24/chb24_17", 3515, 3581],
              ["chb24/chb24_21", 2804, 2872],
              ]
data_files2 = np.array(data_files2)


print("starting to get data from files")

batch_num =0

X= np.empty((0,n_chs,int(seg_size * fs/10)))
Y= np.empty((0))



for i in range(26, 36):  # data_files.shape[0]-split_data): 54
        sez_start = data_files[i, 1]
        # print(sez_start)
        sez_end = data_files[i, 2]
        # print(sez_end)
        file_name = data_files[i, 0]

        # read file by EDFreader
        print("open sez file:", file_name)
        path = "E:/eeg data/chb-mit/" + str(file_name) + ".edf"

        file_name2 = data_files2[i, 0]
        print("open non- sez file:", file_name2)
        path2 = "E:/eeg data/chb-mit/" + str(file_name2) + ".edf"

        hdl = EDFreader(path)
        nst = hdl.getTotalSamples(0)
        ibuf = np.empty(nst, dtype=np.int32)
        dbuf = np.empty(nst, dtype=np.float_)
        datarecord = []

        hdl2 = EDFreader(path2)
        nst2 = hdl2.getTotalSamples(0)
        ibuf2 = np.empty(nst2, dtype=np.int32)
        dbuf2 = np.empty(nst2, dtype=np.float_)
        datarecord2 = []

        for j in [0,3,5,7,9]:#,11,12,15,20,21]:#,, n_chs):
            # print("\nSignal: %s" %(hdl.getSignalLabel(i)))
            # signalnames.append(str(hdl.getSignalLabel(i)))
            # print("Samplefrequency: %f Hz" %(hdl.getSampleFrequency(i)))
            # hdl.rewind(i)
            hdl.readSamples(j, dbuf, hdl.getTotalSamples(j))
            hdl2.readSamples(j, dbuf2, hdl2.getTotalSamples(j))
            # normalize data between 0 and 255

            #ibuf[:] = (255 * (ibuf[:] - np.min(ibuf[:])) / np.ptp(ibuf[:])).astype(int)
            #ibuf2[:] = (255 * (ibuf2[:] - np.min(ibuf2[:])) / np.ptp(ibuf2[:])).astype(int)
            # hdl.readSamples(i, dbuf, hdl.getTotalSamples(i))
            datarecord.append(dbuf.copy())
            datarecord2.append(dbuf2.copy())
        hdl.close()
        hdl2.close()

        datarecord = np.array(datarecord)
        datarecord2 = np.array(datarecord2)
        data1 = datarecord2[:, 0:int(3 * 60*fs)]
        data1 = sk.preprocessing.normalize(data1)
        # data1 = data1 / np.linalg.norm(data1)
        if (int(sez_start) - 10 * 60) * int(fs)>0:
            data2 = datarecord[:, (int(sez_start) - 3 * 60) * int(fs):(int(sez_start) - 0 * 60) * int(fs)]
        else:
            data2 = datarecord[:, (int(sez_start) - 3 * 60) * int(fs):(int(sez_start) - 0 * 60) * int(fs)]
        data2 = sk.preprocessing.normalize(data2)
        # data2 = data2 / np.linalg.norm(data2)

        data3 = datarecord[:, (int(sez_start) - 0 * 60) * int(fs):(int(sez_end) - 0 * 60) * int(fs)]
        # data3 = data3 / np.linalg.norm(data3)

        data = np.concatenate((data1, data2), axis=1)

        # plt.plot(data[3, :])

        targets = np.zeros(int(data.shape[1] / 256 / seg_size))
        targets[int(data.shape[1] / 256 / seg_size / 2):] = 1

        data_set_ = []
        for kk in range(0, data.shape[1], seg_size * int(fs)):
            data_set_.append(data[:, kk:kk + seg_size * int(fs)])

        data_set_ = np.array(data_set_)

        data_set = np.zeros([data_set_.shape[0], data_set_.shape[1], int(data_set_.shape[2] / 10)])
        for i1 in range(0, data_set_.shape[0]):
            for i2 in range(0, data_set_.shape[1]):
                data_set[i1, i2] = data_set_[i1][i2].reshape(-1, 10).mean(axis=1)


        X = np.concatenate((X,data_set), axis=0)
        Y = np.concatenate((Y,targets), axis=0)
        batch_num += 1


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 5

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train)).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
"""
train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
"""


def init_w(seed):
    #return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)
    return tf.keras.initializers.glorot_normal(seed=seed)
    #return tf.keras.initializers.Orthogonal(gain=0.5, seed=seed)
# prepare model

def rnd_num():
    return np.random.randint(1, 50)


def get_compiled_model(input_shape):
    global model
    # Make net

    inputs = keras.layers.Input(shape=input_shape)
    x = inputs[:,:,:,0]
    x = inputs  # [:, :, :, tf.newaxis]

    x = keras.layers.Conv2D(1024, kernel_size=[3, 7], kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            padding='same',
                            kernel_initializer=init_w(rnd_num()))(x)
    x = LeakyReLU()(x)

    x = keras.layers.MaxPooling2D([1, 2], [1, 2], padding='same')(x)

    x = keras.layers.Conv2D(712, kernel_size=[3, 7], kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            padding='same',
                            kernel_initializer=init_w(rnd_num()))(x)
    x = LeakyReLU()(x)
    x = keras.layers.MaxPooling2D([1, 2], [1, 2], padding='same')(x)

    x = keras.layers.Conv2D(512, kernel_size=[3, 7], kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            padding='same',
                            kernel_initializer=init_w(rnd_num()))(x)
    x = LeakyReLU()(x)

    x = keras.layers.MaxPooling2D([1, 2], [1, 2], padding='same')(x)

    x = keras.layers.Conv2D(315, kernel_size=[3, 7], kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            padding='same',
                            kernel_initializer=init_w(rnd_num()))(x)
    x = LeakyReLU()(x)
    x = keras.layers.MaxPooling2D([2, 2], [2, 2], padding='same')(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, kernel_size=[3, 7], kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            padding='same',
                            kernel_initializer=init_w(rnd_num()))(x)
    x = LeakyReLU()(x)
    x = keras.layers.MaxPooling2D([2, 4], [2, 4], padding='same')(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, kernel_size=[3, 7], kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            padding='same',
                            kernel_initializer=init_w(rnd_num()))(x)
    x = LeakyReLU()(x)
    x = keras.layers.MaxPooling2D([2, 4], [2, 4], padding='same')(x)
    # x = keras.layers.BatchNormalization()(x)

    # x = keras.layers.concatenate([ch1_layer7,ch2_layer7])
    x = keras.layers.Flatten()(x)

    # x = keras.layers.TimeDistributed()(x)
    # x = keras.layers.LSTM(32, activation='tanh', return_sequences=True)(x)

    x = keras.layers.Dense(32, activation='sigmoid', kernel_initializer=init_w(rnd_num()))(x)
    # x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(16, activation='sigmoid', kernel_initializer=init_w(rnd_num()))(x)

    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, decay=1e-5, epsilon=0.01, clipnorm=1, clipvalue=1),
        # loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        # metrics=[keras.metrics.SparseCategoricalAccuracy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    # keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    print(model.summary())

    return model


# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./new_age3_ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_or_restore_model(input_shape):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    model = get_compiled_model(input_shape)
    plot_model(model, to_file='new_age3.png', show_shapes=True, show_layer_names=True)
    return model


# make model
with tf.device("GPU:0"):
    model = make_or_restore_model(input_shape)

# set callbackes for fit
callbacks = [
    # This callback saves a SavedModel every epoch
    # We include the current epoch in the folder name.
    tf.keras.callbacks.CSVLogger("new_age3_log.csv", separator=",", append=True),
    keras.callbacks.EarlyStopping(patience=3, monitor="val_binary_accuracy", restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
          patience=5, min_lr=0.001),
    # keras.callbacks.TensorBoard(log_dir='./march5new_logs', histogram_freq=1,
    #                            write_graph=True, update_freq="epoch",
    #                            write_images=True),
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/new_age3_ckpt-file.h5", save_best_only=True,
        monitor="val_loss")
]

# create csv for test results
if not os.path.exists("./new_age3_test_results.csv"):
    print("create new file for test results")
    with open('./new_age3_test_results.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['loss', 'acc'])




#model.load_weights("E:\\work\\python\\EDFlib-Python-master\\new_age_ckpt1\\chb10-wights.h5")
print("Training start")
model.fit(train_dataset, validation_data=val_dataset, batch_size=64, epochs=130, verbose=1, callbacks=callbacks, shuffle=False)

# mask = np.isnan(data)
# data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])

y = model.predict(x=X_test, batch_size=32)

print(classification_report(y_true=y_test, y_pred=y.round()))

cm = confusion_matrix(y_true=y_test, y_pred=y.round())

sns.heatmap(cm, annot=True, fmt='g', xticklabels=['inter-ictal', 'pre-ictal'], yticklabels=['inter-ictal', 'pre-ictal'])

false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y)
print('roc_auc_score: ', roc_auc_score(y_test, y))


plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


import math,time

###########
def alpha(iter_num):
    return rnd_num() * math.exp(-iter_num)

def alpha2(iter_num):
    return rnd_num() + math.exp(-iter_num/math.pi)

for j in range(0,50):
    iter_num = 0
    chg_layers = 18
    for i in range(0,18):
        iter_num +=10
        chg_layers -=1
        model_weights = np.array(model.get_weights())         # save weights in a np.array of np.arrays

        model_weights[0:chg_layers] = model_weights[0:chg_layers] * alpha(iter_num)

        model.set_weights(model_weights)                  # add 1 to all weights in the neural network
        model.fit(train_dataset, validation_data=val_dataset, batch_size=64, epochs=130, verbose=1, callbacks=callbacks,
              shuffle=False)
    """
    cont = input("Continue train?")
    if cont == '0':
        break
    else:
        continue
    """