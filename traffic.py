import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import time
import pickle
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5] [*]")


    # Get image arrays and labels for all image files
    # If data file exists, read from it. Else, call the load_data(dir) method
    if os.path.isfile('data.pickle'):
        with open('data.pickle', 'rb') as f:
            data = pickle.load(f)
            images, labels = data
    else:
        images, labels = data = load_data(sys.argv[1])
        with open('data2.pickle', 'wb') as f:
            pickle.dump(data, f)

    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    
    # Use saved model for predictions on test data 
    if os.path.isfile('model.h5'):
        model = tf.keras.models.load_model(sys.argv[2])
        model.evaluate(x_test, y_test, verbose=2)
        return


    model = get_model()
    model.fit(x_train, y_train, epochs=EPOCHS)
    model.evaluate(x_test,  y_test, verbose=2)

    
    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")



def load_data(data_dir):
    images = []
    labels = []
    dir_labels = os.listdir(data_dir)
    for label in dir_labels:
        if label != ".DS_Store": #Added because gtsrb-small has a .DS_Store file
            combined_path = os.path.join(data_dir, label)
            images_for_label = os.listdir(combined_path)
            for i in range(len(images_for_label)):
                labels.append(label)
            for img in images_for_label:
                img_path = os.path.join(combined_path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image,(IMG_WIDTH, IMG_HEIGHT))
                images.append(image)

    return (images, labels)
    

def get_model():
    # Creating a CNN
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(
            64, (4, 4), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
    )

    
    return model


if __name__ == "__main__":
    main()
