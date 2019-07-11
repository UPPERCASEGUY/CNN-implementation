#keras contains many great image processing datasets such as mnist,cifar10.... 
import keras
#loading mnist dataset(handwritten digits' images) . 
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()



#reshape data to fit model(bcz our model needs 4 dim matrices as input)
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
#we have 60000 images to train and 10000 images to test


#making evey neuron value lie b/w 0 and 1(dividing by max value).MIMP STEP
X_train = X_train/255
X_test =X_test/255

#converting the digits into binary array,
#e.g. -> 3 represented as [0,0,0,1,0,0,0,0,0,0]
from keras.utils import to_categorical
#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train[0] #the representation of 0th ouput digit



# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout
#CNN IMPLEMENTATION
#create model
model = Sequential()
#add model layers(our model will output the reduced image with less parameters ....which will be our input for fully connected part of the neural network)

#convolution using 100 filters(kernels),kernel size = 5x5x1 ,relu function as an activator,input shape as (28,28,1)
model.add(Conv2D(100, kernel_size=5, activation='relu', input_shape=(28,28,1)))

#max pooling of 2x2 submatrices
model.add(MaxPooling2D(pool_size=(2,2)))

#convolution using 100 filters(kernels),kernel size = 5x5x1 ,relu function as an activator,input shape as (28,28,1)
model.add(Conv2D(100, kernel_size=5, activation='relu'))

#max pooling of 2x2 submatrices
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
#joining the reduced image to the fully connected network.
model.add(Dense(10, activation='softmax'))




#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(X_train.shape)
#train the model
#fitting our model to train using 10 epochs.
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)







#models predected probabilities for first 4 images.
model.predict(X_test[:4])
#actual results for first 4 images in test set
y_test[:4]