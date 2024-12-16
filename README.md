AI Solar Prediction Model
This project provides a web-based interface to predict solar power generation using advanced AI techniques. The model uses weather data as input and provides hourly solar generation predictions.

Web App Details
Features:
Interactive Web Interface: Enter weather data, select prediction duration, and view results.
Hourly Solar Power Forecasting: Realistic hourly generation patterns based on diurnal variations.
Dynamic Graphs: Line plots visualizing predictions for durations ranging from 1 day to 2 months.
Export Options: Download predictions as a CSV file.
Input Parameters:
Weather Inputs: Temperature, Wind Speed, Wind Direction, Humidity, Air Pressure, Cloud Cover, Ground Temperature.
Prediction Duration: Options include 1 day, 2 days, 1 week, 1 month, etc.
Output:
Visual graph of hourly predictions.
Predicted data available for download.
Plugin Details
Features:
Seamless Integration: Easily integrate the AI model into other applications via API endpoints.
Realistic Patterns: Hourly predictions account for sunrise, sunset, and peak daylight.
Endpoints:
Home: GET /
Displays the home page.

Input Page: GET /input
Form to input weather parameters and select prediction duration.

Prediction: POST /predict_input
Returns hourly solar generation predictions as JSON or renders a graph on the results page.

Running the App
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Start the server:
bash
Copy code
python app.py
Access the app in your browser:
arduino
Copy code
http://127.0.0.1:5000
License
This project is open-source and licensed under the MIT License.
