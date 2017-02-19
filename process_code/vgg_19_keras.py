from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

def VGG_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    print model.output_shape
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    print model.output_shape
    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    print model.output_shape
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print model.output_shape

    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    print model.output_shape
    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    print model.output_shape
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print model.output_shape

    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    print model.output_shape
    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    print model.output_shape
    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    print model.output_shape
    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    print model.output_shape
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print model.output_shape

    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    print model.output_shape
    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    print model.output_shape
    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    print model.output_shape
    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    print model.output_shape
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print model.output_shape

    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    print model.output_shape
    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    print model.output_shape
    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    print model.output_shape
    model.add(ZeroPadding2D((1,1)))
    print model.output_shape
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    print model.output_shape
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print model.output_shape

    model.add(Flatten())
    print model.output_shape
    model.add(Dense(4096, activation='relu'))
    print model.output_shape
    model.add(Dropout(0.5))
    print model.output_shape
    model.add(Dense(4096, activation='relu'))
    print model.output_shape
    model.add(Dropout(0.5))
    print model.output_shape
    model.add(Dense(1000, activation='softmax'))
    print model.output_shape

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    im = cv2.resize(cv2.imread('dog.jpg'), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = VGG_19('../models/vgg19_weights.h5')

    print 'Finish loading the model'
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    print np.argmax(out)