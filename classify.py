import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
<<<<<<< HEAD
from tensorflow.keras.layers.experimental import preprocessing
    
    
def classify(img, model):
    model = tf.keras.models.load_model(model)
    
=======

def classify(img, model):
    # Load the model
    model = tf.keras.models.load_model(model)
>>>>>>> 703f9b07132fcc7bea2394199cf0a99188d7f391
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    # #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability


# # initialize the input image shape (224x224 pixels) along with
# # the pre-processing function (this might need to be changed
# # based on which model we use to classify our image)
# inputShape = (224, 224)
# preprocess = imagenet_utils.preprocess_input
# # if we are using the InceptionV3 or Xception networks, then we
# # need to set the input shape to (299x299) [rather than (224x224)]
# # and use a different image pre-processing function
# if network in ("Inception", "Xception"):
#     inputShape = (299, 299)
#     preprocess = preprocess_input

# Network = MODELS[network]
# model = Network(weights="imagenet")

# # load the input image using PIL image utilities while ensuring
# # the image is resized to `inputShape`, the required input dimensions
# # for the ImageNet pre-trained network
# image = Image.open(BytesIO(bytes_data))
# image = image.convert("RGB")
# image = image.resize(inputShape)
# image = img_to_array(image)
# # our input image is now represented as a NumPy array of shape
# # (inputShape[0], inputShape[1], 3) however we need to expand the
# # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# # so we can pass it through the network
# image = np.expand_dims(image, axis=0)
# # pre-process the image using the appropriate function based on the
# # model that has been loaded (i.e., mean subtraction, scaling, etc.)
# image = preprocess(image)

# preds = model.predict(image)
# predictions = imagenet_utils.decode_predictions(preds)
# imagenetID, label, prob = predictions[0][0]