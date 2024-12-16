# AI Solar Prediction Model

This project provides a web-based interface to predict solar power generation using advanced AI techniques. The model uses weather data as input and provides hourly solar generation predictions.

---
## Datsets likns:
**Solar power-generation data**
(DATA.GO.KR) https://www.data.go.kr/data/15099650/fileData.do 
Period: 2022-04-01 ~ 2024-06-30 (39,360 ea, over 2 years)
(Datetime) Year, Month, Date, hour, minutes, second
(Location) Plant name, lat, lng
(Generation) Power-generation value(Wh)

**Weather data**
(KMA) https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36
Period: 2022-04-01 ~ 2024-06-30
(Temp) Temperature, Ground temperature
(Solar) Irradiance(hr), Radiation(MJ/m2)
(Etc) Humidity, Amount of cloud, Wind direction, Wind speed

## Web App Details

### Features
- **Interactive Web Interface**: Enter weather data, select prediction duration, and view results.
- **Hourly Solar Power Forecasting**: Realistic hourly generation patterns based on diurnal variations.
- **Dynamic Graphs**: Line plots visualizing predictions for durations ranging from 1 day to 2 months.
- **Export Options**: Download predictions as a CSV file.

### Input Parameters
- **Weather Inputs**: 
  - Temperature
  - Wind Speed
  - Wind Direction
  - Humidity
  - Air Pressure
  - Cloud Cover
  - Ground Temperature
- **Prediction Duration**: 
  - 1 Day
  - 2 Days
  - 1 Week
  - 1 Month
  - Custom durations up to 2 months

### Output
- **Graph**: Visualizes hourly solar power generation predictions.
- **Downloadable CSV**: Predictions available for export.

---

## Plugin Details
**Python 3.8 or later**
**Flask**
**PyTorch**
**Numpy**
**Pandas**
**Scikit-learn**
**Chart.js (for frontend visualization)**
### Features
- **Seamless Integration**: Easily integrate the AI model into other applications via API endpoints.
- **Realistic Patterns**: Hourly predictions account for sunrise, sunset, and peak daylight hours.

### API Endpoints
1. **Home**: `GET /`  
   Displays the home page.
   
2. **Input Page**: `GET /input`  
   Form to input weather parameters and select prediction duration.
   
3. **Prediction**: `POST /predict_input`  
   Returns hourly solar generation predictions as JSON or renders a graph on the results page.

---

## Running the App

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/solar-prediction-model.git
   cd solar-prediction-model
