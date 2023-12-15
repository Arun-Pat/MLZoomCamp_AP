
import os
import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image

''' IMPORTANT
    Only one of the following imports must be uncommented.
    Docker: tflite_runtime
    Local testing: tensorflow.lite
'''

import tensorflow.lite as tflite # Comment this when using it wth Lambda while making docker file
# import tflite_runtime.interpreter as tflite # Comment this when testing the pedict function locally

MODEL_NAME = os.getenv('MODEL_NAME', 'vegetable_classification_model.tflite')

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def prepare_input(x):
    return x / 255.0


interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']



def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(224, 224))

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
   
    return preds
    # return float(preds[0, 0])

def predict_names(url, N=1):  # N = How many Top predicted categories to display
    output_data = predict(url)
    classes = ['Bean','Bitter_Gourd','Bottle_Gourd','Brinjal','Broccoli','Cabbage','Capsicum',
                'Carrot', 'Cauliflower', 'Cucumber','Papaya','Potato','Pumpkin','Radish','Tomato']

    # Get the indices of the top N values in the output_data
    top_indices = np.argsort(output_data)[0, ::-1][:N]
    # Map the indices to the corresponding category labels
    top_categories = [classes[i] for i in top_indices]
    return top_categories


def lambda_handler(event, context):
    url = event['url']
    top_cat = event['top_cat']    # How many Top predicted categories to display

    pred = predict_names(url, top_cat)
    
    result = {
        'prediction': pred
    }
    return result

if __name__ == "__main__":
    image_url = "https://1.bp.blogspot.com/-o3BArH9Lq1M/XPm3DJcx8aI/AAAAAAAAJSA/_zzUjuGx9k8-lTu0B9hYvtOVogOYEch7ACLcBGAs/s1600/p7_Broccoli_HH1812_gi905351392.jpg"
    # pred = predict(image_url)
    # print(pred)
    N = 2
    predictions =  predict_names(image_url, N)
    print("Top {} predicted categories:".format(N), predictions)
