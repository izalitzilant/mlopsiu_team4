from flask import Flask, request, jsonify, abort, make_response

import mlflow
import mlflow.pyfunc
import os
import pandas as pd

BASE_PATH = os.path.expandvars("$PROJECTPATH")

model = mlflow.pyfunc.load_model(os.path.join(BASE_PATH, "api", "model_dir"))

app = Flask(__name__)

@app.route("/info", methods = ["GET"])
def info():
	response = make_response(str(model.metadata), 200)
	response.content_type = "text/plain"
	return response

@app.route("/", methods = ["GET"])
def home():
	msg = """
	Welcome to our ML service to predict Advertisement Demand for Avito Platform\n\n

	This API has two main endpoints:\n
	1. /info: to get info about the deployed model.\n
	2. /predict: to send predict requests to our deployed model.\n

	"""

	response = make_response(msg, 200)
	response.content_type = "text/plain"
	return response

@app.route("/predict", methods = ["POST"])
def predict():
	if not request.json:
		abort(400)

	data = request.json
	data = data["inputs"]
	meta = model.metadata.signature
	meta_inputs = meta.inputs.to_dict()
	data = pd.DataFrame.from_dict(data, orient="index").T
	for col in meta_inputs:
		col_name = col['name']
		if col_name not in data.columns:
			abort(400, f"Missing {col_name} in inputs")
		col_type = col['type']
		try:
			if col_type == "long":
				data[col_name] = data[col_name].astype(int)
			elif col_type == "double":
				data[col_name] = data[col_name].astype(float)
		except:
			abort(400, f"Invalid type for {col_name}. Expected {col_type}. Cannot convert {data[col_name]} to {col_type}")

	predictions = model.predict(data).flatten()
	response = make_response(jsonify(predictions.tolist()), 200)
	response.content_type = "application/json"
	return response


# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)