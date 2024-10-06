from flask import Flask, request, jsonify, render_template, g

import joblib
import numpy as np
import logging
import time
from apscheduler.schedulers.background import BackgroundScheduler
import os

app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model
model = joblib.load('model.pkl')

@app.before_request
def start_timer():
    g.start = time.time()

@app.after_request
def log_request(response):
    if 'start' in g:
        diff = time.time() - g.start
        app.logger.info(f'{request.method} {request.path} - {response.status_code} - {diff:.4f}s')
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json(force=True)

        # Extract input values from JSON
        order_month = data.get('order_month')
        order_year = data.get('order_year')
        price_ratio = data.get('price_ratio')
        rating = data.get('rating')

        # Prepare input data for prediction
        input_data = np.array([order_month, order_year, price_ratio, rating]).reshape(1, -1)

        # Make prediction using the model
        prediction = model.predict(input_data)

        # Log the request and prediction
        logging.info(f'Received data: {data}')
        logging.info(f'Prediction: {prediction[0]}')

        # Return the prediction as a JSON response
        return jsonify({'predicted_sales_amount': prediction[0]})

    except Exception as e:
        # Log the error
        logging.error(f'Error: {e}')
        return jsonify({'error': str(e)})

@app.route('/logs', methods=['GET'])
def logs():
    try:
        with open('app.log', 'r') as log_file:
            log_data = log_file.read()
        return render_template('logs.html', log_data=log_data.splitlines())
    except Exception as e:
        logging.error(f'Error reading log file: {e}')
        return jsonify({'error': str(e)})

# Custom error handler for 404 errors
@app.errorhandler(404)
def not_found(error):
    return "", 404  # Return an empty response with 404 status code

if __name__ == '__main__':
    app.run(debug=True)
