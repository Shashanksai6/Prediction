#!pip install flask-ngrok
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model('/Users/vicky/Documents/output.h5')

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def hello():
    return 'Hello from Flask!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file)

    prepared_image = prepare_image(image, target_size=(224, 224))

    predictions = model.predict(prepared_image)

    # Assuming your model outputs probabilities
    tumor_probability = predictions[0][0]

    # Define a threshold probability above which we consider it a positive prediction
    threshold = 0.5

    if tumor_probability > threshold:
        prediction_text = 'Yes, there is a tumor.'
    else:
        prediction_text = 'No, there is no tumor.'

    response = {
        'prediction': prediction_text
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()
