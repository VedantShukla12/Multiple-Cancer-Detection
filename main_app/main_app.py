from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Mapping model names to their corresponding ports
MODEL_PORTS = {
    "lung": 5001,
    "breast": 5002,
    "prostate": 5003
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/redirect", methods=["POST"])
def redirect_to_model():
    model = request.form.get("model")
    if model not in MODEL_PORTS:
        return "Invalid model selected", 400

    # Redirect to the corresponding model server
    return redirect(f"http://localhost:{MODEL_PORTS[model]}/", code=302)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000, debug=True)
