import csv
import cv2
import argparse
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense
from keras.layers import Conv2D, Lambda, MaxPooling2D, Dropout, Activation, Convolution2D, Cropping2D, BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2

from keras.optimizers import Adam


class Pipeline:

    def __init__(self, model=None, base_path='', finetune_path='',  epochs=20):
        self.data = []
        self.model = model
        self.epochs = epochs
        self.training_samples = []
        self.validation_samples = []
        self.correction_factor = 0.2
        self.base_path = base_path
        self.image_path = self.base_path + '/IMG/'
        self.finetune_path = finetune_path
        self.driving_log_path = self.base_path + '/driving_log.csv'
        self.finetune_image_path = self.finetune_path + '/IMG/'
        self.finetune_driving_log_path = self.finetune_path + '/driving_log.csv'

    def import_driving_log_data(self, train=True):
        if train:
            path = self.driving_log_path
        else:
            path = self.finetune_driving_log_path
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                self.data.append(line)

    def generate_multi_images(self, sample, train=True, flip=True, multi_cam=True):

        images, measurements = [], []
        measurement = np.float32(sample[3])

        if multi_cam:
            for i in range(3):
                image_name = sample[i].split('/')[-1]
                if train:
                    image = cv2.imread(self.image_path + image_name)
                    image = image[50:140,:,:]
                    image = cv2.resize(image,(200, 66), interpolation = cv2.INTER_AREA)
                else:
                    image = cv2.imread(self.finetune_image_path + image_name)
                    image = image[50:140,:,:]
                    image = cv2.resize(image,(200, 66), interpolation = cv2.INTER_AREA)
                # left image
                if i == 1:
                    measurement = measurement + self.correction_factor
                # right image
                elif i == 2:
                    measurement = measurement - self.correction_factor
                images.append(image)
                measurements.append(measurement)
                if flip:
                    images.append(cv2.flip(image, 1))
                    measurements.append(-1.0 * measurement)
    

        else:
            image_name = sample[0].split('/')[-1]
            if train:
                image = cv2.imread(self.image_path + image_name)
                image = image[50:140,:,:]
                image = cv2.resize(image,(200, 66), interpolation = cv2.INTER_AREA)
            else:
                image = cv2.imread(self.finetune_image_path + image_name)
                image = image[50:140,:,:]
                image = cv2.resize(image,(200, 66), interpolation = cv2.INTER_AREA)
            images.append(image)
            measurements.append(measurement)    
            if flip:
                images.append(cv2.flip(image, 1))
                measurements.append(-1.0 * measurement)    
        
        return images, measurements


    def data_generator(self, samples, flip, multi_cam, train=True, batch_size=128):
        num_samples = len(samples)

        while True:
            samples = shuffle(samples)

            for offset in range(0, num_samples, batch_size):
                end = offset + batch_size
                batch_samples = samples[offset: end]
                images, measurements = [], []

                for sample in batch_samples:
                    multi_images, multi_measurements = self.generate_multi_images(sample, train=train, flip=flip, multi_cam=multi_cam)
                    images.extend(multi_images)
                    measurements.extend(multi_measurements)

                X_train, y_train = np.array(images), np.array(measurements)
                yield shuffle(X_train, y_train)

    def split_data(self):
        train, validation = train_test_split(self.data, test_size=0.2)
        self.training_samples, self.validation_samples = train, validation


    def train_generator(self, train=True, batch_size=128):
        return self.data_generator(samples=self.training_samples, flip=True, multi_cam=True, train=train, batch_size=batch_size)

    def validation_generator(self, train=True, batch_size=128):
        return self.data_generator(samples=self.validation_samples, flip=True, multi_cam=False, train=train, batch_size=batch_size)

    
    def setup_to_finetune(self):
        for layer in self.model.layers[:-5]:
            layer.trainable = False
        for layer in self.model.layers:
            print(layer, layer.trainable)

    def run(self):
        
        print("Importting driving log data...")
        self.import_driving_log_data(train=True)
        print("Spliting data...")
        self.split_data()
        print("training data size: ", len(self.data))
        self.model.fit_generator(generator=self.train_generator(),
                                 validation_data=self.validation_generator(),
                                 nb_epoch=self.epochs,
                                 samples_per_epoch=25600,
                                 nb_val_samples=len(self.validation_samples))
        
        self.model.save('model01.h5')
        
        self.model = load_model('model01.h5')
        self.model.summary()
        # Finetune with new data
        self.import_driving_log_data()
        self.split_data()
        print("fine tune data size: ", len(self.data))
        #self.setup_to_finetune()
        self.model.fit_generator(generator=self.train_generator(),
                                 validation_data=self.validation_generator(),
                                 nb_epoch=20,
                                 samples_per_epoch=25600,
                                 nb_val_samples=len(self.validation_samples))
        self.model.save('model02.h5')

        


def Network(loss='mse', optimizer=Adam(lr=0.0001)):
    model = Sequential()
    # preprocessing the image
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
    # conv layer 1
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), W_regularizer=l2(0.001),  dim_ordering="tf"))
    model.add(BatchNormalization())
    model.add(ELU())
     # conv layer 2
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2),  W_regularizer=l2(0.001),  dim_ordering="tf"))
    model.add(BatchNormalization())
    model.add(ELU())
    # conv layer 3
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2),  W_regularizer=l2(0.001), dim_ordering="tf"))
    model.add(BatchNormalization())
    model.add(ELU())
    # conv layer 4
    model.add(Convolution2D(64, 3, 3,  W_regularizer=l2(0.001),  dim_ordering="tf"))
    model.add(BatchNormalization())
    model.add(ELU())
    # conv layer 5
    model.add(Convolution2D(64, 3, 3,  W_regularizer=l2(0.001), dim_ordering="tf"))
    model.add(BatchNormalization())
    model.add(ELU())
    # flatten layer
    model.add(Flatten())
    # fully connected layer 1
    model.add(Dense(100,  W_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(ELU())
    # fully connected layer 2
    model.add(Dense(50,  W_regularizer=l2(0.001)))
    model.add(ELU())
    # fully connected layer 3
    model.add(Dense(10,  W_regularizer=l2(0.001)))
    model.add(ELU())
    # output layer
    model.add(Dense(1))
    model.summary()
    # compile the model
    model.compile(loss=loss, optimizer=optimizer)

    return model

def Network2():
    model = Sequential()

    # Normalize
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))
    model.add(Cropping2D(cropping=((70, 25), (0,0))))
    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001),  dim_ordering="tf"))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001),  dim_ordering="tf"))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001),  dim_ordering="tf"))
    model.add(ELU())

    #model.add(Dropout(0.50))
    
    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001),  dim_ordering="tf"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001),  dim_ordering="tf"))
    model.add(ELU())

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Dense(50, W_regularizer=l2(0.001)))

    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))

    # Add a fully connected output layer
    model.add(Dense(1))

    # Compile and train the model, 
    #model.compile('adam', 'mean_squared_error')
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model

def main():


    pipeline = Pipeline(model=Network2(), base_path='/home/mengling/Desktop/data', finetune_path='/home/mengling/Desktop/train07')
    pipeline.run()

if __name__ == '__main__':
    main()
    

