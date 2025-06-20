from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (ensure the .h5 file is in the same directory or provide the path)
MODEL_PATH = "lung_model.h5"
model = load_model(MODEL_PATH)

# Define the class labels (extracted from your ImageDataGenerator)
class_labels = {0: 'Lung squamous_cell_carcinoma', 1: 'Lung adenocarcinoma', 2: 'Lung benign tissue'}

# Define a function to preprocess images before prediction
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image / 256.0  # Normalize as needed

# Define the route for the home page
@app.route("/", methods=["GET"])
def index():
    # Display the upload form
    return render_template("lung.html")

# Define the route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the file from the POST request
        file = request.files["file"]
        if not file:
            return "No file uploaded!", 400
        
        # Save the file locally (optional) and load the image
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        # Preprocess the image
        image = load_img(file_path, target_size=(256, 256))  # Adjust the size to match your model's input
        image = preprocess_image(image, target_size=(256, 256))
        
        # Make predictions
        preds = model.predict(image)
        
        # Assuming softmax output, get the class with the highest prediction probability
        predicted_class = np.argmax(preds, axis=1)[0]
        predicted_label = class_labels.get(predicted_class, "Unknown")

        # Get the prediction percentage for each class
        prediction_percentages = preds[0] * 100  # Convert to percentage
        
        # Return the result as a JSON response
        return jsonify({
            "prediction": predicted_label,
            "percentages": {class_labels[i]: f"{prediction_percentages[i]:.2f}%" for i in range(len(class_labels))}
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5001, debug=True)
