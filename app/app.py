from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route("/", methods=["GET", "POST"])
def inputs():
    if request.method == "POST":
        smile = request.values.get("inputSmile")
        result = {
            "prediction": smile,
        }
        return render_template("results.html", result=result)
    return render_template("inputs.html")


@app.route("/predict", methods=["POST"])
def predict():
    return """Hello World"""


if __name__ == "__main__":
    app.run(debug=True)