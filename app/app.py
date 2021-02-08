from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

from utils import smile2bytes, Prediction

app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route("/", methods=["GET", "POST"])
def inputs():
    if request.method == "POST":
        model = request.form["options"]
        inputs = request.values.get("inputSmile")
        prediction = Prediction(model=model, inputs=inputs)
        res = prediction.predict()
        print(res)
        return render_template("results.html", result=res)
    return render_template("inputs.html")


@app.route("/predict", methods=["POST"])
def predict():
    return """Hello World"""


if __name__ == "__main__":
    app.run(debug=True)