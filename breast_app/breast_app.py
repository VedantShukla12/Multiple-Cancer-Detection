from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

print("Breast model found:", os.path.exists("model_breast.h5"))

# Upload folder setup
app.config['BREAST_UPLOAD_FOLDER'] = "static/breast_uploads"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load model
breast_model = tf.keras.models.load_model("model_breast.h5")
print("✅ Breast model loaded.")

# Class labels
breast_labels = {0: 'Cancerous', 1: 'Non-Cancerous'}

# Utilities
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_from_path(image_path, target_size):
    image = load_img(image_path, target_size=target_size)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image / 255.0

# Home routes
@app.route("/")
def index():
    return render_template("breast.html")#analyzebreast.html was b4

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/analyzebreast")
def analyzebreast():
    return render_template("breast.html")

# Breast Prediction
@app.route('/predict/breast', methods=['POST'])
def predict_breast():
    if 'image' not in request.files:
        return jsonify({'error': "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': "No image selected"}), 400

    if file and allowed_file(file.filename):
        os.makedirs(app.config['BREAST_UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['BREAST_UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)

        img = preprocess_image_from_path(filepath, (256, 256))
        preds = breast_model.predict(img)
        predicted_class = np.argmax(preds, axis=1)[0]
        prediction_percentages = preds[0] * 100

        return jsonify({
            'result': breast_labels[predicted_class],
            'probability': float(preds[0][predicted_class]),
            'image_url': f"/{filepath.replace(os.sep, '/')}",
            'percentages': {
                breast_labels[i]: f"{prediction_percentages[i]:.2f}%" for i in range(len(breast_labels))
            }
        })

    return jsonify({'error': "Invalid file type"}), 400

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5002, debug=True)
