import tensorflow as tf 
import numpy as np
tf.random.set_seed(5)
np.random.seed(5)

# load the data 
data = tf.keras.datasets.mnist.load_data()

(X_train_full,y_train_full), (X_test,y_test) = data

X_train, X_val = X_train_full[5000:], X_train_full[:5000]
y_train, y_val = y_train_full[5000:], y_train_full[:5000]

X_train_preprocessed = X_train[...,np.newaxis] / 255
X_val_preprocessed = X_val[...,np.newaxis] / 255

# chossing model architecture 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.compile(
    loss = tf.keras.losses.sparse_categorical_crossentropy,
    metrics = [tf.keras.metrics.sparse_categorical_accuracy],
    optimizer = tf.keras.optimizers.Adam(),
)

# learning rate automatic reduction during training  
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_sparse_categorical_accuracy', 
    patience=3, 
    verbose=1, 
    factor=0.5, 
    min_lr=0.00001,
)

# ganerating more data from images
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train_preprocessed)

# training
history = model.fit_generator(
    datagen.flow(X_train_preprocessed,y_train, batch_size=32),
    epochs=30,
    validation_data=(X_val_preprocessed,y_val),
    callbacks=[
        # tf.keras.callbacks.TensorBoard(log_dir = f'mylogdir'),
        learning_rate_reduction,
    ],
)

# saving
model.save("static/model.hdf5")