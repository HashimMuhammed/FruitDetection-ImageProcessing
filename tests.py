# USAGE
# python test_network.py --model santa_not_santa.model --image examples/bbb.jpg

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import operator
"""
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
"""

def do_processing(path_to_file):
    # load the image
    image = cv2.imread(path_to_file)
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model("MODEL")

    """
    # classify the input image
    (notSanta, santa) = model.predict(image)[0]

    # build the label
    label = "Santa" if santa > notSanta else "Not Santa"
    proba = santa if santa > notSanta else notSanta
    label = "{}: {:.2f}%".format(label, proba * 100)
    """

        # classify the input image
    (Unknown, Orange, Apple, Avocado,Banana,Guava,Papaya,Pineapple,Watermelon) = model.predict(image)[0]

        # build the label
    label_dict = {
            "Orange": Orange,
            "Apple": Apple,
            "Avocado":Avocado,
            "Banana":Banana,
            "Guava":Guava,
            "Papaya":Papaya,
            "Pineapple":Pineapple,
             "Watermelon": Watermelon,
            "Unknown": Unknown
            }

    label = max(label_dict.items(), key=operator.itemgetter(1))[0]
    max_value = label_dict[label]

    label = "{}: {:.2f}%".format(label, max_value * 100)

    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

    # show the output image
    cv2.imwrite("output.jpg", output)
    cv2.waitKey(0)
