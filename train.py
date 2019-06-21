# -*- coding: UTF-8 -*
from tensorflow import keras
batch_size = 64

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.,
                        shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

# flow_from_directory
# train contains two documents : cat and dog
# validation contains two documents : cat and dog
train_generator = train_datagen.flow_from_directory('./dataset/train/',
                                                target_size=(224, 224),
                                                batch_size=batch_size,
                                                class_mode='categorical')

validation_generator = val_datagen.flow_from_directory('./dataset/validation/',
                                                target_size=(224, 224),
                                          batch_size=batch_size, class_mode='categorical')


base_model = keras.applications.ResNet50(input_tensor=keras.layers.Input(
                        (224, 224, 3)), weights=None ,include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
x = keras.layers.Dense(4096, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(4096,activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
prediction = keras.layers.Dense(2, activation='softmax')(x)
model = keras.models.Model(base_model.input, prediction)

# We don't train the ResNet50
for layer in base_model.layers:
   layer.trainable = False

lrate = 0.01
decay = lrate/100
sgd = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Callback for loss logging per epoch
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.losses.append(logs.get('val_loss'))
history = LossHistory()

# Callback for early stopping in the training
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                               patience=4, verbose=0, mode='auto')
checkpointer = keras.callbacks.ModelCheckpoint(filepath='./weight/cat-dog-weights.hdf5',
                                               verbose=1, save_best_only=True)

fitted_model = model.fit_generator(train_generator,
                                   # batches of samples =  sample_nums // batch_size
                                   steps_per_epoch= len(train_generator) // batch_size,
                                   epochs=1000,
                                   validation_data=validation_generator,
                                   validation_steps= len(validation_generator) // batch_size,
                                   callbacks=[early_stopping, history,checkpointer],
                                   verbose=1)
model.save('./ResNet50-17-modelResNet50.h5')

