from flask import Flask, render_template, request
from predict import predict_answer
import base64

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    image_file = request.files["image"]
    question = request.form["question"]

    image_bytes = image_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    answer = predict_answer(image_bytes, question)

    return render_template(
        "index.html",
        image_data=image_base64,
        question=question,
        answer=answer
    )

if __name__ == "__main__":
    app.run(debug=True)