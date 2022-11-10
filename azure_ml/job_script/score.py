import os
import logging
import json
import numpy
# import joblib
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        "./", "model/data/model/"
    )
    # deserialize the model file back into a sklearn model
    # model = joblib.load(model_path)
    model = tf.keras.models.load_model(model_path)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    logging.info("model 1: request received")
    data = json.loads(raw_data)["data"]
    data = base64.b64decode(data)
    img = Image.open(io.BytesIO(data))
    out = []
    out.append(np.asarray(img))
    out = np.asarray(out)
    data = out.reshape(out.shape[0], 28, 28, 1).astype('float32')
    result = model.predict(data)
    result = tf.squeeze(result).numpy()
    predicted_ids = np.argmax(result, axis=-1)
    predicted_class_names = classes[predicted_ids]
    print(predicted_class_names)
    logging.info("Request processed")
    return predicted_class_names

"""
    init()
    import base64
    test_image_url = '../../dataset/test/2/3.jpg'
    data = {}
    with open(test_image_url, mode='rb') as file:
        img = file.read()
    data['data'] = base64.b64encode(img).decode('utf-8')
    print(data)
    body = str.encode(json.dumps(data))
    print(run(body))
"""