import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator


from subprocess import check_output
print(check_output(["ls","../input/digit-recognizer"]).decode("utf8"))

# Load train and test data
train = pd.read_csv("../input/digit-recognizer/train.csv")
print(train.shape)
train.head()

test = pd.read_csv("../input/digit-recognizer/test.csv")
print(test.shape)
test.head()

X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')


# Data Visualization
X_train = X_train.reshape(X_train.shape[0], 28, 28)

for i in range(6, 9) :
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i])
    
# Expand one or more dimension for colour channel gray

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_train.shape



X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_test.shape


# Feature Standardization
# Important preprocessing step. Used to center data around zero mean and unit variance
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x) :
    return (x-mean_px)/std_px
# y_train.shape

# One hot encoding for labels 3 => [0,0,0,1,0,0,0,0,0,0]
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
num_classes = y_train.shape[1]
num_classes


#  plotting 4th label
plt.title(y_train[3])
plt.plot(y_train[3])
plt.xticks(range(10));



# Design Neural Network Architecture

# fix random seed for reproducibility
seed = 43
np.random.seed(seed)

# Define Model

model = Sequential()
model.add(Lambda(standardize, input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

print("Input Shape : ",model.input_shape)
print("Output Shape : ",model.output_shape)
model.compile(optimizer=RMSprop(lr=0.001),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
from keras.preprocessing import image
gen = image.ImageDataGenerator()


# Cross Validation

X = X_train
y = y_train

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches = gen.flow(X_val, y_val, batch_size=64)


history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3,
                             validation_data=val_batches, validation_steps=val_batches.n)
history_dict = history.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values)+1)

plt.plot(epochs,loss_values,'bo')
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
plt.clf # clear plot

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(acc_values)+1)

plt.plot(epochs,acc_values,'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

# Lets train a fully connected model to increase accuracy

def get_fc_model() :
    model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model


fc = get_fc_model()
fc.optimizer.lr=0.01
history = fc.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 
                          validation_data=val_batches, validation_steps=val_batches.n)
                          
# Lets train a fully connected model to increase accuracy

def get_fc_model() :
    model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model


fc = get_fc_model()
fc.optimizer.lr=0.01
history = fc.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 
                          validation_data=val_batches, validation_steps=val_batches.n)
                          
# Data Augmentation

gen =  ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches = gen.flow(X_val, y_val, batch_size=64)

cnn.optimizer.lr=0.001
history = cnn.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 
                          validation_data=val_batches, validation_steps=val_batches.n)
                          
                          
                          
                          
                          
                          
                          
                          
                          
# Adding batch normalization
# bn helps to fine tune hyperparameters more better and train really deep networks

from keras.layers.normalization import BatchNormalization


def get_bn_model() :
    model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Convolution2D(32,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32,(3,3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3),activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
        
    ])
    
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = get_bn_model()
model.optimizer.lr=0.01

history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=2, 
                          validation_data=val_batches, validation_steps=val_batches.n)
                          
                          
                          
# Use complete Data to train before submitting

model.optimizer.lr=0.01
gen = image.ImageDataGenerator()
batches = gen.flow(X,y, batch_size=64)
history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3)

predictions = model.predict_classes(X_test, verbose=0)

submissions = pd.DataFrame({"ImageId" : list(range(1, len(predictions)+1)),
                           "Label" : predictions})

submissions.to_csv("PS.csv", index=False, header=True)
