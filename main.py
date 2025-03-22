import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import ipywidgets as widgets
from IPython.display import display, clear_output

# Load the trained model
model = load_model("mnist.h5")

# Function to predict a digit
def predict_digit(image):
    # Resize the image to 28x28
    image = cv2.resize(image, (28, 28))
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values
    image = image.astype("float32") / 255.0
    # Reshape to match model input
    image = image.reshape(1, 28, 28, 1)
    # Predict digit
    predictions = model.predict(image)
    return np.argmax(predictions), max(predictions[0])

# Function to draw a digit and recognize it
def draw_digit():
    print("Upload an image of a digit to recognize it.")

    # Upload widget
    uploader = widgets.FileUpload(accept='image/*', multiple=False)
    display(uploader)

    # Recognition button
    button = widgets.Button(description="Recognize")

    # Output widget
    output = widgets.Output()

    def on_button_clicked(b):
        with output:
            clear_output(wait=True)
            if uploader.value:
                file_info = next(iter(uploader.value.values()))
                content = file_info['content']
                nparr = np.frombuffer(content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Show uploaded image
                cv2_imshow(image)

                # Predict digit
                digit, confidence = predict_digit(image)
                print(f"Predicted Digit: {digit}, Confidence: {confidence*100:.2f}%")
            else:
                print("Please upload an image first.")

    button.on_click(on_button_clicked)
    display(button, output)

# Run the function
draw_digit()
