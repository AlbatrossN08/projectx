from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from statistics import mode

# Load your trained models
with open("final_svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("final_nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)
with open("final_rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Load your symptom index dictionary
with open("data_dict.pkl", "rb") as f:
    data_dict = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_disease():
    try:
        if request.method == "POST":
            # Get symptoms from request body
            symptoms = request.form.get('symptoms')
            symptoms_list = symptoms.split(",")

            # Create input data for models
            input_data = [0] * len(data_dict["symptom_index"])
            for symptom in symptoms_list:
                index = data_dict["symptom_index"].get(symptom)
                if index is not None:
                    input_data[index] = 1

            # Reshape and predict
            input_data = np.array(input_data).reshape(1, -1)
            rf_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
            nb_prediction = data_dict["predictions_classes"][nb_model.predict(input_data)[0]]
            svm_prediction = data_dict["predictions_classes"][svm_model.predict(input_data)[0]]

            # Use mode logic to determine final prediction
            predictions = [rf_prediction, nb_prediction, svm_prediction]
            final_prediction = mode(predictions)
            predictions = {
                "final_prediction": final_prediction,
                }
            return render_template("index.html", predictions=predictions, symptoms=symptoms)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Invalid input or error occurred"}), 400

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        if request.method == "POST":
            data = request.get_json(force=True)  # Get data posted as a json
            if 'symptoms' not in data:
                return jsonify({"error": "Symptoms key not found in request data"}), 400
            
            symptoms = data['symptoms']
            symptoms_list = symptoms.split(",")
            
            # Create input data for models
            input_data = [0] * len(data_dict["symptom_index"])
            for symptom in symptoms_list:
                index = data_dict["symptom_index"].get(symptom)
                if index is not None:
                    input_data[index] = 1

            # Reshape and predict
            input_data = np.array(input_data).reshape(1, -1)
            rf_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
            nb_prediction = data_dict["predictions_classes"][nb_model.predict(input_data)[0]]
            svm_prediction = data_dict["predictions_classes"][svm_model.predict(input_data)[0]]

            # Use mode logic to determine final prediction
            predictions = [rf_prediction, nb_prediction, svm_prediction]
            final_prediction = mode(predictions)

            return jsonify({'final_prediction': final_prediction, 'symptoms': symptoms}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Invalid input or error occurred"}), 400

if __name__ == "__main__":
    app.run(debug=True, port=8089)
