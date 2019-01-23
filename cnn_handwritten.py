# Hand written Digit Recognition with Convolutional Neural Network

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Convolution Layer1
classifier.add(Conv2D(28, (3, 3), input_shape = (28, 28, 3), activation = 'relu'))

# Pooling Layer1
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Convolutional layer2
classifier.add(Conv2D(28, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening Layer
classifier.add(Flatten())

# Fully Connected Layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set/',
                                                 target_size = (28, 28),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set/',
                                            target_size = (28, 28),
                                            batch_size = 10,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 100,
                         epochs = 500,
                         validation_data = test_set,
                         validation_steps = 100)

#Making new predictions

import numpy as np
from keras.preprocessing import image
test_image_1 = image.load_img('dataset/prediction/which_digit_1.jpg', target_size = (28, 28))

test_image = image.img_to_array(test_image_1)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

out=result[0]
output=[]
for i in range(out.size):
	output.append(int(out[i]))

prediction = 'Unpredictable'
for idx,cls in enumerate(output):
	if cls==1:
		prediction=idx
print("Predicted Digit: ",prediction)

