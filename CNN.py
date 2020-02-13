import numpy as np
import matplotlib.pyplot as plt
import split_folders
from sklearn.metrics import confusion_matrix, classification_report
from keras import models, layers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator

# dataset path
input_path = "/Users/shaomengyuan/flowers"
output_path = "/Users/shaomengyuan/assignment"
# file path from draw.py, use a method to split dataset.
train_path = "/Users/shaomengyuan/assignment/train"
val_path = "/Users/shaomengyuan/assignment/val"
test_path = "/Users/shaomengyuan/assignment/test"

batchsize = 25
imagesize = (64, 64)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range= 0.2, 
        zoom_range= 0.2,
        horizontal_flip= True
        )

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    train_path, 
                    target_size=imagesize,
                    class_mode='categorical',
                    batch_size= batchsize
                    )

val_generator = val_datagen.flow_from_directory(
                    val_path, 
                    target_size=imagesize,
                    class_mode = 'categorical',
                    batch_size=batchsize)

test_generator = test_datagen.flow_from_directory(
                    test_path, 
                    target_size=imagesize,
                    class_mode = 'categorical',
                    batch_size= batchsize,
                    shuffle=False)

# Build a model
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', 
            input_shape = (imagesize[0], imagesize[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128, (3,3), Activation = 'relu')
# model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(Dropout(0.5))

model.add(layers.Dense(512, activation = 'sigmoid'))
model.add(layers.Dense(5, activation = 'softmax'))

# compole the model
model.compile(
              optimizer = 'rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy']
              )

# fit the model with train_generator.
history = model.fit_generator(
                    train_generator, 
                #     steps_per_epoch = 100,
                    steps_per_epoch = len(train_generator), 
                    validation_data = val_generator, 
                    validation_steps = len(val_generator),
                #     use_multiprocessing = True,
                #     validation_steps = 60,
                    epochs = 20
                    )

# save the model
model.save('/Users/shaomengyuan/assignment/models.h5')
model.summary()

model_eval = model.evaluate_generator(test_generator, len(test_generator))
print(model_eval)

# predict by the test dataset-- confusion matrix and report the accuracy
predictions = model.predict_generator(
                test_generator, len(test_generator))
t_predict = np.argmax(predictions, axis= 1)
# print(t_predict)
# print(predictions.shape, len(t_predict))
# print(predictions)
t_test = test_generator.classes
# print(t_test)
# print(t_test.shape, len(t_test))
# print(t_predict)
# print(len(t_predict))
print('Confusion Matrix')
cm = confusion_matrix(t_test, t_predict)
print(cm)

# Evaluation accuracy
print('Classification Report')
cr = classification_report(t_test, t_predict)
print(cr)


# visualize the loss and accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'y.', label = 'Training acc')
plt.plot(epochs, val_acc, '.-', label = 'Validation acc')
plt.title('accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'r.', label = 'Training loss')
plt.plot(epochs, val_loss, '.-', label = 'Validation loss')
plt.title('Train and Validation loss and accuracy')
plt.legend()
plt.show()


# pickle.dumps(the model), just save the model as pickle string, 
# pickle.dumps(model, filepath), save the model as pickle in file.

# def main():
#     plot_acc()
#     plot_loss()

# if __name__ == "__main__":
#     main()