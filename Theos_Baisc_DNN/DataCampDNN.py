# Importing the necessary functionality
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import Flatten, MaxPooling2D

def run():
    # Creating the model
    DNN_Model = Sequential()

    # Inputting the shape to the model
    DNN_Model.add(Input(shape = (256, 256, 3)))

    # Creating the deep neural network
    DNN_Model.add(Conv2D(256, (3, 3), activation='relu', padding = "same"))
    DNN_Model.add(MaxPooling2D(2, 2))
    DNN_Model.add(Conv2D(128, (3, 3), activation='relu', padding = "same"))
    DNN_Model.add(MaxPooling2D(2, 2))
    DNN_Model.add(Conv2D(64, (3, 3), activation='relu', padding = "same"))
    DNN_Model.add(MaxPooling2D(2, 2))

    # Creating the output layers
    DNN_Model.add(Flatten())
    DNN_Model.add(Dense(64, activation='relu'))
    DNN_Model.add(Dense(10))

    tf.keras.utils.plot_model(DNN_Model, to_file='plots/model_big.png', show_shapes=True)


if __name__ == "__main__":
    run()
