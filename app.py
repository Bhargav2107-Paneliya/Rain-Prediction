from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)


model = pickle.load(open("C:/Users/bharg/Project/AI/Project/Rain_Pred/model_xgb.pkl", "rb"))
scaler = pickle.load(open("C:/Users/bharg/Project/AI/Project/Rain_Pred/scaler.pkl", "rb"))

features = [
    'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Cloud9am', 'Cloud3pm', 'Temp3pm', 'RainToday'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
        Sunshine = float(request.form.get("Sunshine"))
        WindGustSpeed = float(request.form.get("WindGustSpeed"))
        Humidity9am = float(request.form.get("Humidity9am"))
        Humidity3pm = float(request.form.get("Humidity3pm"))
        Pressure9am = float(request.form.get("Pressure9am"))
        Cloud9am = float(request.form.get("Cloud9am"))
        Cloud3pm = float(request.form.get("Cloud3pm"))
        Temp3pm = float(request.form.get("Temp3pm"))
        RainToday = request.form.get("RainToday").strip().lower()

        RainToday = 1 if RainToday in ["yes", "y", "1"] else 0

        # Create DataFrame
        df = pd.DataFrame([[Sunshine, WindGustSpeed, Humidity9am, Humidity3pm,
                            Pressure9am, Cloud9am, Cloud3pm, Temp3pm, RainToday]],
                          columns=features)

        # Scale input
        scaled_input = scaler.transform(df)

        # Predict
        y_pred = model.predict(scaled_input)[0]
        y_prob = model.predict_proba(scaled_input)[0][1]

        confidence = round(y_prob*100,2)

        if y_pred == 1:
            result = "🌧️ Yes, it will rain tomorrow!"
        else:
            result = "☀️ No rain expected tomorrow."

        return render_template("index.html", prediction_text=result,confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
