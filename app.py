from flask import Flask, request, render_template
import pandas as pd
import torch
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from TrainModelwithMHA_updated import RNNWithAttention, input_size, hidden_size, output_size, num_heads

app = Flask(__name__)

# Load the trained model
model_path = './model/trained_rnn_model_attention_b_7f.pth'
model = RNNWithAttention(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_heads=num_heads)
model.load_state_dict(torch.load(model_path))
model.eval()


# Define routes for home, input, and results pages
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict_input', methods=['POST'])
def predict_input():
    # Collect input data from the form
    input_data = [float(request.form.get(col)) for col in
                  ['temperature', 'windspeed', 'winddirection', 'humidity', 'airpressure', 'amountofcloud',
                   'groundtemperature']]

    # Get prediction duration
    prediction_time = request.form.get("prediction_time")

    # Define the number of rows based on the prediction duration
    time_durations = {
        "1_day": 1,
        "2_days": 2,
        "3_days": 3,
        "5_days": 5,
        "1_week": 7,
        "2_weeks": 14,
        "1_month": 30,
        "2_months": 60
    }
    num_days = time_durations.get(prediction_time, 1)

    # Preprocess the initial input data
    input_data = np.array(input_data).reshape(1, -1)
    scaler = MinMaxScaler()
    input_data_norm = scaler.fit_transform(input_data)
    input_tensor = torch.tensor(input_data_norm, dtype=torch.float32).unsqueeze(1)

    # Generate daily predictions
    daily_predictions = []
    current_input = input_tensor.clone()
    with torch.no_grad():
        for day in range(num_days):
            prediction = model(current_input)
            prediction_value = prediction.squeeze().item()

            # Clip negative values (set to zero) for better visualization
            if prediction_value < 0:
                prediction_value = 0

            daily_predictions.append(prediction_value)

            # Simulate realistic daily changes in the input features
            temp_change = np.random.uniform(-0.5, 0.5)  # Temperature fluctuation between -0.5 to +0.5°C
            wind_speed_change = np.random.uniform(-0.2, 0.2)  # Wind speed fluctuation between -0.2 to +0.2 m/s
            humidity_change = np.random.uniform(-1.0, 1.0)  # Humidity fluctuation between -1.0% to +1.0%
            cloud_cover_change = np.random.uniform(-0.05, 0.05)  # Cloud cover fluctuation between -5% to +5%

            # Update the input data for the next day's prediction
            current_input[0][0][0] += temp_change  # Update temperature
            current_input[0][0][1] += wind_speed_change  # Update wind speed
            current_input[0][0][3] += humidity_change  # Update humidity
            current_input[0][0][5] += cloud_cover_change  # Update cloud cover

    # Generate hourly data based on daily predictions (if 1-5 days selected)
    hourly_predictions = []
    if num_days <= 5:
        for day_index, daily_value in enumerate(daily_predictions):
            for hour in range(24):
                # Generate a diurnal pattern using a sine wave
                daylight_multiplier = math.sin(math.pi * (hour / 24))  # Peaks at midday (12:00)
                hourly_value = daily_value * max(0, daylight_multiplier)  # Night hours (0-6, 18-23) will have near zero values
                hourly_predictions.append(hourly_value)

        # Prepare data for hourly x-axis
        hours = list(range(1, num_days * 24 + 1))  # 1 to num_days * 24 for the x-axis
        result_data = {'Hour': hours, 'Predicted Generation (Wh)': hourly_predictions}

    else:
        # For more than 5 days, show daily predictions
        result_data = {'Day': list(range(1, num_days + 1)), 'Predicted Generation (Wh)': daily_predictions}

    return render_template('result.html', results=result_data)  # Pass either hourly or daily data to the result page

if __name__ == '__main__':
    app.run(debug=True)