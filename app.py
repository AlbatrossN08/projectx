from flask import Flask, request, jsonify
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

@app.route("/predict", methods=["POST"])
def predict_disease():
    try:
        # Get symptoms from request body
        symptoms = request.json["symptoms"]
        symptoms_list = symptoms.split(",")

        # Create input data for models
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms_list:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1

        # Reshape and predict
        input_data = np.array(input_data).reshape(1, -1)
        rf_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
        nb_prediction = data_dict["predictions_classes"][nb_model.predict(input_data)[0]]
        svm_prediction = data_dict["predictions_classes"][svm_model.predict(input_data)[0]]

        # Calculate final prediction and return response
        final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])
        predictions = {
            "rf_model_prediction": rf_prediction,
            "naive_bayes_prediction": nb_prediction,
            "svm_model_prediction": svm_prediction,
            "final_prediction": final_prediction,
        }

        return jsonify(predictions), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Invalid input or error occurred"}), 400

if __name__ == "__main__":
    app.run(debug=True ,port="8050")
