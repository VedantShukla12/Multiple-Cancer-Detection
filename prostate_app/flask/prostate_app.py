import tensorflow as tf
import numpy as np
import PIL.Image
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename  # Although we don't need to save uploads anymore

app = Flask(__name__)

# --- 1.  Define Paths (Adjust these to your actual file locations) ---
MODEL_PATH = "panda_model.h5"  # Replace with your model path.
IMAGE_DIR = "panda-tiles/versions/3/train"  # <--this owrks

# --- 2.  Constants (Match these to the notebook!) ---
IMG_DIM = (1536, 128)  # Likely not needed if we're not resizing directly
CLASSES_NUM = 6
N = 12

# --- 3. Load Model (Load it ONCE at app startup) ---
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    print("Model loaded successfully (metrics and optimizer ignored)!")

    # Recompile the model with the original settings
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    print("Model recompiled successfully")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set model to None if loading fails


# --- 4. Preprocessing Function (Modified) ---
def preprocess_tiles(image_id, image_dir):
    """Loads, preprocesses, and stacks the 12 tiles for a given image ID from the existing directory."""
    fnames = [image_id + '_' + str(i) + '.png' for i in range(N)]
    images = []
    for fn in fnames:
        img_path = os.path.join(image_dir, fn)
        try:
            img = PIL.Image.open(img_path).convert('RGB')  # Open and convert to RGB
            img = np.array(img)  # Convert to NumPy array
            img = img / 255.0  # Normalize to [0, 1]
            images.append(img)
        except FileNotFoundError:
            print(f"Error: File not found: {img_path}")
            return None  # Or handle the missing tile appropriately

    if len(images) != N:
        print(f"Error: Found {len(images)} tiles instead of {N} for image {image_id}")
        return None

    result = np.stack(images).reshape(1, 1536, 128, 3)  # Shape: (1, 1536, 128, 3)
    return result


# --- 5. Prediction Function ---
def predict_isup_grade(model, image_id, image_dir):
    """Predicts the ISUP grade for a given image ID."""
    if model is None:
        return "Error: Model not loaded.  Check server logs."

    preprocessed_image = preprocess_tiles(image_id, image_dir)
    if preprocessed_image is None:
        return "Error: Could not preprocess image"

    prediction = model.predict(preprocessed_image)  # Make the prediction
    print(f"Raw Prediction: {prediction}")  # Print raw prediction for debugging

    predicted_class = np.argmax(prediction) 
    
    if predicted_class >=1:
        ans = "Malignant Prostate Cancer"
    else:
        ans =  "Benign Prostate Cancer"

    return f"Predicted ISUP Grade: {predicted_class}, {ans}"


# --- 6. Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        # Get the image ID from the form
        image_id = request.form['image_id']

        # Make prediction
        if model:
            prediction = predict_isup_grade(model, image_id, IMAGE_DIR)
        else:
            prediction = "Error: Model not loaded.  Please check server logs."

        return render_template('result.html', prediction=prediction, image_id=image_id)

    return render_template('upload.html')


# --- 7.  Flask Error Handling ---
@app.errorhandler(500)
def internal_error(error):
    return "500 error: {}".format(error)

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


# --- 8. Main Execution (Only runs when script is executed directly) ---
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=5003)  # Enable debug mode for easier development