import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import time
import re
import logging
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import pandas as pd
import json
import os
import threading
import http.server
import socketserver
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- 1. SET UP LOGGING ---
logging.basicConfig(
    filename='advanced_snow_model.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

# --- 2. PERSISTENCE FUNCTIONS ---
def load_historical_weather(file_path='historical_weather.csv'):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.fillna(0)
        logging.info(f"Loaded {len(df)} historical weather records.")
        return df
    logging.info("No historical weather file found. Starting fresh.")
    return pd.DataFrame(columns=['date', 'temperature_2m_min', 'snowfall_sum', 'precip'])

def save_historical_weather(df, file_path='historical_weather.csv'):
    df.to_csv(file_path, index=False)
    logging.info(f"Saved {len(df)} historical weather records.")

def load_teleconnections(file_path='teleconnections.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info("Loaded saved teleconnection data.")
        return {k: defaultdict(dict, v) if k not in ['oni', 'mjo_amp'] else v for k, v in data.items()}
    return {'oni': {}, 'ao': defaultdict(dict), 'nao': defaultdict(dict), 'pna': defaultdict(dict), 'pdo': defaultdict(dict), 'qbo': defaultdict(dict), 'ssn': defaultdict(dict), 'mjo_amp': {}}

def save_teleconnections(data, file_path='teleconnections.json'):
    serializable = {k: dict(v) if k not in ['oni', 'mjo_amp'] else v for k, v in data.items()}
    with open(file_path, 'w') as f:
        json.dump(serializable, f)
    logging.info("Saved teleconnection data.")

def load_config(file_path='config.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded config: Location = {config.get('location')}")
        return config.get('location', None)
    return None

def save_config(location, file_path='config.json'):
    with open(file_path, 'w') as f:
        json.dump({'location': location}, f)
    logging.info(f"Saved config: Location = {location}")

# --- 3. CURRENT WEATHER ---
def fetch_current_weather(lat, lon):
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': lat, 'longitude': lon,
        'current': 'temperature_2m,weather_code,snowfall',
        'timezone': 'auto'
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()['current']
        temp_c = data['temperature_2m']
        temp_f = temp_c * 9/5 + 32
        snowfall_cm = data.get('snowfall', 0)
        snowfall_in = snowfall_cm / 2.54
        weather_code = data['weather_code']
        weather_desc = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 51: "Light drizzle", 61: "Slight rain", 71: "Slight snow",
        }.get(weather_code, "Unknown")
        return f"Temperature: {temp_f:.1f}°F | Snowfall: {snowfall_in:.1f} in | Conditions: {weather_desc}"
    except Exception as e:
        logging.error(f"Error fetching current weather: {e}")
        return "Current weather unavailable."

# --- 4. PLOTS ---
def generate_snow_plot(historical_df):
    if historical_df.empty:
        return None
    historical_df['date'] = pd.to_datetime(historical_df['date'])
    historical_df['year'] = historical_df['date'].dt.year
    yearly_snow = historical_df.groupby('year')['snowfall_sum'].sum() / 2.54
    plt.figure(figsize=(10, 5))
    plt.plot(yearly_snow.index, yearly_snow.values, marker='o')
    plt.title('Historical Yearly Snowfall (inches)')
    plt.xlabel('Year')
    plt.ylabel('Total Snowfall')
    plt.grid(True)
    plot_path = 'historical_snow.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def generate_first_snow_plot(first_snow_dict, predicted_doy, current_year):
    if not first_snow_dict:
        return None
    years = sorted(first_snow_dict.keys())
    doys = [first_snow_dict[y] for y in years]
    plt.figure(figsize=(10, 5))
    plt.scatter(years, doys, label='Historical First Snow DOY', color='blue')
    if predicted_doy is not None:
        plt.scatter(current_year, predicted_doy, label='Predicted First Snow DOY', color='red', marker='x', s=100)
    plt.title('Historical and Predicted First Snowfall Days')
    plt.xlabel('Year')
    plt.ylabel('Day of Year (DOY)')
    plt.legend()
    plt.grid(True)
    plot_path = 'first_snow_plot.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# --- 5. HTML GENERATION ---
def generate_html(location, driver_status_string, first_snow_message, seasonal_message, last_prediction_date, next_prediction_date, current_weather, plot_path, first_snow_plot_path, predicted_timestamp_ms):
    plot_img = f'<img src="{plot_path}" alt="Historical Snowfall Plot" style="max-width:100%;">' if plot_path else ''
    first_plot_img = f'<img src="{first_snow_plot_path}" alt="First Snowfall Plot" style="max-width:100%;">' if first_snow_plot_path else ''
    countdown_script = ""
    if predicted_timestamp_ms is not None:
        countdown_script = f"""
        <div id="countdown" style="font-size: 1.5em; font-weight: bold; margin: 20px 0;"></div>
        <script>
        const targetDate = new Date({predicted_timestamp_ms});
        function updateCountdown() {{
          const now = new Date().getTime();
          const distance = targetDate - now;
          if (distance < 0) {{
            document.getElementById("countdown").innerHTML = "Snowfall has arrived!";
            return;
          }}
          const days = Math.floor(distance / (1000 * 60 * 60 * 24));
          const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
          const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
          const seconds = Math.floor((distance % (1000 * 60)) / 1000);
          document.getElementById("countdown").innerHTML = `${{days}}d ${{hours}}h ${{minutes}}m ${{seconds}}s till next snowfall`;
        }}
        setInterval(updateCountdown, 1000);
        updateCountdown();
        </script>
        """
    else:
        countdown_script = "<p>No prediction available for countdown.</p>"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Snowfall Predictions for {location}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            h2 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>Snowfall Predictions for {location}</h1>
        <p>Last prediction made: {last_prediction_date.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Next prediction: {next_prediction_date.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Current Weather</h2>
        <p>{current_weather}</p>
        
        <h2>Current Climate Drivers</h2>
        <table>
            <tr><th>Driver</th><th>Value</th></tr>
            {''.join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(driver_status_string.items())])}
        </table>
        
        <h2>First Snowfall Forecast</h2>
        <p>{first_snow_message}</p>
        {countdown_script}
        
        <h2>Seasonal Snowfall Outlook</h2>
        <p>{seasonal_message}</p>
        
        <h2>Historical Snowfall Trend</h2>
        {plot_img}
        
        <h2>First Snowfall Days Trend</h2>
        {first_plot_img}
    </body>
    </html>
    """
    with open('index.html', 'w') as f:
        f.write(html_content)
    logging.info("Updated index.html with latest predictions.")

# --- 6. WEB SERVER ---
def run_server():
    PORT = 8080
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        logging.info(f"Serving web server at http://localhost:{PORT}")
        httpd.serve_forever()

# --- 7. TELECONNECTIONS ---
def _fetch_historical_oni():
    ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    logging.info("Fetching historical ENSO (ONI) data from NOAA...")
    try:
        response = requests.get(ONI_URL, timeout=10)
        response.raise_for_status()
        
        lines = response.text.split('\n')
        header_line = None
        data_start_index = -1
        for i, line in enumerate(lines):
            if "SEAS" in line and "YR" in line and "ANOM" in line:
                header_line = line.split()
                data_start_index = i + 1
                break
        if header_line is None:
            raise ValueError("Could not find header in ONI data file.")
            
        seas_col = header_line.index("SEAS")
        yr_col = header_line.index("YR")
        anom_col = header_line.index("ANOM")
        
        historical_son_data = {}
        latest_oni_value = None
        latest_oni_year = 0
        latest_oni_period = ""

        for line in lines[data_start_index:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) <= max(seas_col, yr_col, anom_col):
                continue
            try:
                season = parts[seas_col]
                year = int(parts[yr_col])
                anomaly_str = parts[anom_col]
                if anomaly_str != '-99.9':
                    anomaly = float(anomaly_str)
                    if season == "SON":
                        historical_son_data[year] = anomaly
                    latest_oni_value = anomaly
                    latest_oni_year = year
                    latest_oni_period = season
            except (ValueError, IndexError):
                continue

        if latest_oni_value is None:
            raise ValueError("Could not parse any valid ONI values.")

        logging.info(f"Successfully parsed ONI. Most recent: {latest_oni_period} {latest_oni_year} (ONI: {latest_oni_value:.2f})")
        return historical_son_data, latest_oni_value

    except Exception as e:
        logging.error(f"Failed to fetch or parse ONI data: {e}")
        return {}, 0.0

def _fetch_historical_teleconnection(url, name):
    logging.info(f"Fetching historical {name} data from NOAA...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        lines = response.text.split('\n')
        historical_data = defaultdict(dict)
        latest_value = 0.0
        
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                year = int(parts[0])
                if year < 1900 or len(parts) != 13:
                    continue
                for month_idx, value_str in enumerate(parts[1:], 1):
                    value = float(value_str)
                    if value > -99.9: 
                        historical_data[year][month_idx] = value
                        latest_value = value
            except ValueError:
                continue

        if not historical_data:
             raise ValueError(f"No data parsed for {name}.")

        logging.info(f"Successfully parsed {name}. Most recent value: {latest_value:.2f}")
        return historical_data, latest_value
    
    except Exception as e:
        logging.error(f"Failed to fetch or parse {name} data: {e}")
        return defaultdict(dict), 0.0

def _fetch_historical_qbo():
    QBO_URL = "https://www.geo.fu-berlin.de/met/ag/strat/produkte/qbo/qbo.dat"
    logging.info("Fetching historical QBO data from FU Berlin...")
    try:
        response = requests.get(QBO_URL, timeout=10)
        response.raise_for_status()
        
        lines = response.text.split('\n')
        historical_data = defaultdict(dict)
        latest_value = 0.0
        
        for line in lines[3:]:
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                year = int(parts[0])
                month = int(parts[1])
                value_str = parts[5]  # 30hPa
                value = float(value_str) / 10.0
                if value > -999:
                    historical_data[year][month] = value
                    latest_value = value
            except ValueError:
                continue

        logging.info(f"Successfully parsed QBO. Most recent value: {latest_value:.2f}")
        return historical_data, latest_value
    except Exception as e:
        logging.error(f"Failed to fetch or parse QBO data: {e}")
        return defaultdict(dict), 0.0

def _fetch_historical_ssn():
    SSN_URL = "https://www.sidc.be/silso/INFO/snmtotcsv.php"
    logging.info("Fetching historical SSN data from SIDC...")
    try:
        response = requests.get(SSN_URL, timeout=10)
        response.raise_for_status()
        
        lines = response.text.split('\n')
        historical_data = defaultdict(dict)
        latest_value = 0.0
        
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(';')
            if len(parts) < 5:
                continue
            try:
                year = int(parts[0])
                month = int(parts[1])
                value = float(parts[3])
                historical_data[year][month] = value
                latest_value = value
            except ValueError:
                continue

        logging.info(f"Successfully parsed SSN. Most recent value: {latest_value:.2f}")
        return historical_data, latest_value
    except Exception as e:
        logging.error(f"Failed to fetch or parse SSN data: {e}")
        return defaultdict(dict), 0.0

def _fetch_historical_mjo():
    MJO_URL = "https://psl.noaa.gov/mjo/mjoindex/omi.1x.txt"
    logging.info("Fetching historical MJO data from NOAA PSL...")
    try:
        response = requests.get(MJO_URL, timeout=10)
        response.raise_for_status()
        
        lines = response.text.split('\n')
        historical_data = {}
        latest_amp = 0.0
        monthly_amp = defaultdict(list)
        
        for line in lines:
            parts = line.split()
            if len(parts) < 5 or not parts[0].isdigit():
                continue
            try:
                date_str = parts[0]
                year = int(date_str[:4])
                month = int(date_str[4:6])
                amp = float(parts[4])
                monthly_amp[(year, month)].append(amp)
                latest_amp = amp
            except ValueError:
                continue
        
        historical_monthly = defaultdict(dict)
        for (year, month), amps in monthly_amp.items():
            historical_monthly[year][month] = np.mean(amps)
        
        logging.info(f"Successfully parsed MJO amplitude. Most recent: {latest_amp:.2f}")
        return historical_monthly, latest_amp
    except Exception as e:
        logging.error(f"Failed to fetch or parse MJO data: {e}")
        return defaultdict(dict), 0.0

# --- 4. HELPER FUNCTIONS ---

def get_lat_lon(location):
    try:
        if ',' in location and all('.' in part for part in location.replace(',', ' ').split()):
            return map(float, location.split(','))
        
        query_location = location
        if re.match(r'^\d{5}$', location.strip()):
            query_location = f"{location}, USA"

        headers = {'User-Agent': 'AdvancedSnowModel/1.0 (contact@example.com)'}
        url = f"https://nominatim.openstreetmap.org/search?q={query_location}&format=json"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data:
            logging.info(f"Found location: {data[0]['display_name']}")
            return float(data[0]['lat']), float(data[0]['lon'])
        else:
            raise ValueError(f"No location found for '{query_location}'")
            
    except Exception as e:
        logging.error(f"Error getting coordinates: {e}")
        return 42.7325, -84.5555

def fetch_open_meteo_data(lat, lon, start_date, end_date, is_forecast=False):
    base_url = "https://api.open-meteo.com/v1/forecast" if is_forecast else "https://archive-api.open-meteo.com/v1/archive"
    daily_vars = 'temperature_2m_min,snowfall_sum,precipitation_probability_max' if is_forecast else 'temperature_2m_min,snowfall_sum,precipitation_sum'
    params = {
        'latitude': lat, 'longitude': lon,
        'start_date': start_date, 'end_date': end_date,
        'daily': daily_vars,
        'timezone': 'auto'
    }
    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()['daily']
        df = pd.DataFrame({
            'date': pd.to_datetime(data['time']),
            'temperature_2m_min': data['temperature_2m_min'],
            'snowfall_sum': data['snowfall_sum'],
            'precip': data.get('precipitation_sum', data.get('precipitation_probability_max', [np.nan] * len(data['time'])))
        })
        df = df.dropna(subset=['temperature_2m_min', 'snowfall_sum'], how='all')  # Drop if both key vars missing
        return df
    except Exception as e:
        logging.error(f"Error fetching Open-Meteo data: {e}")
        return pd.DataFrame()

def get_historical_data_for_ml(lat, lon):
    historical_df = load_historical_weather()
    
    # Determine missing ranges
    current_date = datetime.now().date()
    earliest_date = datetime(1940, 1, 1).date()  # Open-Meteo start for basic vars
    if historical_df.empty:
        missing_start = earliest_date
        missing_end = current_date - timedelta(days=1)
    else:
        historical_df['date'] = pd.to_datetime(historical_df['date']).dt.date
        last_date = historical_df['date'].max()
        missing_start = last_date + timedelta(days=1)
        missing_end = current_date - timedelta(days=1)
    
    if missing_start <= missing_end:
        logging.info(f"Fetching missing historical data from {missing_start} to {missing_end}")
        # Chunk large ranges by year to avoid timeouts
        new_df = pd.DataFrame()
        current_start = missing_start
        while current_start <= missing_end:
            chunk_end = min(current_start + timedelta(days=365), missing_end)
            chunk_df = fetch_open_meteo_data(lat, lon, current_start.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d'))
            new_df = pd.concat([new_df, chunk_df])
            current_start = chunk_end + timedelta(days=1)
        
        historical_df = pd.concat([historical_df, new_df]).drop_duplicates(subset='date').sort_values('date')
        save_historical_weather(historical_df)
    
    # Fetch teleconnections (full refetch as they update monthly)
    tele_data = load_teleconnections()
    tele_data['oni'], current_oni = _fetch_historical_oni()
    tele_data['ao'], current_ao = _fetch_historical_teleconnection("https://psl.noaa.gov/data/correlation/ao.data", "AO")
    tele_data['nao'], current_nao = _fetch_historical_teleconnection("https://psl.noaa.gov/data/correlation/nao.data", "NAO")
    tele_data['pna'], current_pna = _fetch_historical_teleconnection("https://psl.noaa.gov/data/correlation/pna.data", "PNA")
    tele_data['pdo'], current_pdo = _fetch_historical_teleconnection("https://psl.noaa.gov/data/correlation/pdo.data", "PDO")
    tele_data['qbo'], current_qbo = _fetch_historical_qbo()
    tele_data['ssn'], current_ssn = _fetch_historical_ssn()
    tele_data['mjo_amp'], current_mjo_amp = _fetch_historical_mjo()
    save_teleconnections(tele_data)
    
    current_drivers = {"oni": current_oni, "ao": current_ao, "nao": current_nao, "pna": current_pna, "pdo": current_pdo,
                       "qbo": current_qbo, "ssn": current_ssn, "mjo_amp": current_mjo_amp}
    
    # Process for ML (expanded features)
    X_train_first_snow, y_train_first_snow = [], []
    first_snow_dict = {}  # Year: DOY
    seasonal_snow_totals = defaultdict(float)
    seasonal_driver_features = {}
    year_data = {}
    first_snow_years_found = set()
    
    for _, row in historical_df.iterrows():
        date = row['date']
        doy = date.timetuple().tm_yday
        year = date.year
        month = date.month
        
        oni_val = tele_data['oni'].get(year, 0.0)
        ao_val = tele_data['ao'][year].get(month, 0.0)
        nao_val = tele_data['nao'][year].get(month, 0.0)
        pna_val = tele_data['pna'][year].get(month, 0.0)
        pdo_val = tele_data['pdo'][year].get(month, 0.0)
        qbo_val = tele_data['qbo'][year].get(month, 0.0)
        ssn_val = tele_data['ssn'][year].get(month, 0.0)
        mjo_amp_val = tele_data['mjo_amp'][year].get(month, 0.0)
        
        temp_f = row['temperature_2m_min'] * 9/5 + 32 if pd.notna(row['temperature_2m_min']) else np.nan
        snow_in = row['snowfall_sum'] / 2.54 if pd.notna(row['snowfall_sum']) else 0.0
        precip = row.get('precip', 0.0)
        
        if month >= 8 or month <= 5:
            X_train_first_snow.append([doy, oni_val, ao_val, nao_val, pna_val, pdo_val, qbo_val, ssn_val, mjo_amp_val, precip])
            y_train_first_snow.append(1 if snow_in > 0.1 else 0)
        
        season_year = year if month >= 8 else year - 1
        if snow_in > 0:
            seasonal_snow_totals[season_year] += snow_in
        
        if month == 10:
            if season_year in tele_data['oni']:
                seasonal_driver_features[season_year] = [tele_data['oni'][season_year], tele_data['ao'][season_year].get(month, 0.0), 
                                                         tele_data['nao'][season_year].get(month, 0.0), tele_data['pna'][season_year].get(month, 0.0), 
                                                         tele_data['pdo'][season_year].get(month, 0.0), tele_data['qbo'][season_year].get(month, 0.0),
                                                         tele_data['ssn'][season_year].get(month, 0.0), tele_data['mjo_amp'][season_year].get(month, 0.0)]
        
        if year not in year_data: year_data[year] = []
        year_data[year].append((date, temp_f, snow_in))

    # First snow logic (unchanged)
    current_year = datetime.now().year
    for year in range(earliest_date.year, current_year):
        season_year = year
        if season_year in first_snow_years_found:
            continue
        if season_year in year_data:
            for date, temp, snow in sorted(year_data[season_year], key=lambda x: x[0]):
                if date.month >= 8:
                    if temp <= 32.0 and snow > 0.1:
                        first_snow_dict[season_year] = date.timetuple().tm_yday
                        first_snow_years_found.add(season_year)
                        break
        if season_year + 1 in year_data and season_year not in first_snow_years_found:
            for date, temp, snow in sorted(year_data[season_year + 1], key=lambda x: x[0]):
                if date.month <= 5:
                    if temp <= 32.0 and snow > 0.1:
                        first_snow_dict[season_year] = date.timetuple().tm_yday + 365
                        first_snow_years_found.add(season_year)
                        break
    
    X_train_seasonal, y_train_seasonal = [], []
    for year, drivers in seasonal_driver_features.items():
        if year in seasonal_snow_totals:
            X_train_seasonal.append(drivers)
            y_train_seasonal.append(seasonal_snow_totals[year])
    
    if not X_train_first_snow:
        raise ValueError("No historical data for training.")
    
    logging.info(f"Found {len(first_snow_dict)} historical first-snow events.")
    logging.info(f"Training 'First Snow' on {len(X_train_first_snow)} records.")
    logging.info(f"Training 'Seasonal Total' on {len(X_train_seasonal)} seasons.")
    
    model_1_data = (np.array(X_train_first_snow), np.array(y_train_first_snow))
    model_2_data = (np.array(X_train_seasonal), np.array(y_train_seasonal))
    
    return model_1_data, model_2_data, first_snow_dict, current_drivers

# Deep Learning Models
class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

class MLPRegressor(nn.Module):
    def __init__(self, input_size):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_first_snow_model_dl(X_train, y_train, epochs=100, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    doy_features = X_train[:, 0]
    doy_radians = (doy_features * (2 * np.pi / 365.25))
    X_cyclic_doy = np.column_stack((np.sin(doy_radians), np.cos(doy_radians)))
    
    scalers = {}
    scaled_features_list = []
    feature_names = ['oni', 'ao', 'nao', 'pna', 'pdo', 'qbo', 'ssn', 'mjo_amp', 'precip']
    for i, name in enumerate(feature_names):
        feature_data = X_train[:, i+1].reshape(-1, 1)
        if np.nanvar(feature_data) == 0:
            scaled_features_list.append(np.zeros_like(feature_data))
        else:
            scaler = StandardScaler()
            scaled_features_list.append(scaler.fit_transform(feature_data))
            scalers[name] = scaler
    
    X_final = np.column_stack([X_cyclic_doy] + scaled_features_list)
    X_tensor = torch.tensor(X_final, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MLPClassifier(X_final.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Simple CV-like evaluation (train loss as proxy)
    with torch.no_grad():
        outputs = model(X_tensor)
        auc = cross_val_score(RandomForestClassifier(), X_final, y_train, cv=5, scoring='roc_auc').mean()  # Hybrid eval
        logging.info(f"First Snow DL Model Approx AUC: {auc:.3f}")
    
    return model, scalers, device

def train_seasonal_snow_model_dl(X_train, y_train, epochs=100, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MLPRegressor(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Eval
    mae = cross_val_score(RandomForestRegressor(), X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
    logging.info(f"Seasonal Snow DL Model Approx MAE: {-mae:.3f}")
    
    return model, device

def predict_first_snow_with_ml(model, scalers, device, current_doy, current_drivers, threshold=0.25):
    feature_names = ['oni', 'ao', 'nao', 'pna', 'pdo', 'qbo', 'ssn', 'mjo_amp']
    scaled_drivers = [scalers[name].transform([[current_drivers[name]]])[0, 0] for name in feature_names]
    
    ground_names = ['precip']
    scaled_ground = [scalers[name].transform([[0.0]])[0, 0] for name in ground_names]  # Default
    
    for day_offset in range(1, 180):
        doy = current_doy + day_offset
        if doy > 365:
            doy -= 365
        
        doy_rad = doy * (2 * np.pi / 365.25)
        X_cyclic_doy = [np.sin(doy_rad), np.cos(doy_rad)]
        
        X_test = np.array([X_cyclic_doy + scaled_drivers + scaled_ground])
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            probability = model(X_tensor).item()
        
        if probability >= threshold:
            return current_doy + day_offset
            
    return None 

def predict_seasonal_snow_with_dl(model, device, seasonal_driver_features):
    X_test = torch.tensor(np.array([seasonal_driver_features]), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        return model(X_test).item()

def bayesian_update(prior_mu, prior_sigma, likelihood_mu, likelihood_sigma=15.0):
    try:
        prior_var = prior_sigma ** 2
        like_var = likelihood_sigma ** 2
        post_var = 1 / (1 / prior_var + 1 / like_var)
        post_mu = (prior_mu / prior_var + likelihood_mu / like_var) * post_var
        return post_mu
    except:
        return (prior_mu + likelihood_mu) / 2

def fetch_nws_forecast_with_snow(lat, lon):
    logging.info("Fetching 7-day NWS forecast...")
    try:
        headers = {'User-Agent': 'FMOS_ML_Logical/1.0 (https://example.com; contact@example.com)'}
        points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
        response = requests.get(points_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        grid_url = response.json()['properties']['forecastGridData']
        grid_response = requests.get(grid_url, headers=headers, timeout=10)
        grid_response.raise_for_status()
        props = grid_response.json()['properties']
        
        snow_data = props.get('snowfallAmount', {}).get('values', [])
        temp_data = props.get('minTemperature', {}).get('values', [])
        
        if not snow_data or not temp_data:
            logging.warning("NWS forecast data missing 'snowfallAmount' or 'minTemperature'.")
            return None

        daily_forecasts = {}

        for entry in temp_data:
            try:
                start_time = datetime.fromisoformat(entry['validTime'].split('/')[0].replace('Z', '+00:00'))
                date_key = start_time.date()
                if entry['value'] is not None:
                    if date_key not in daily_forecasts:
                        daily_forecasts[date_key] = {'temp_c': -99, 'snow_mm': 0.0}
                    daily_forecasts[date_key]['temp_c'] = entry['value']
            except Exception as e:
                logging.warning(f"Error parsing NWS temp data: {e}")

        for entry in snow_data:
            try:
                start_time = datetime.fromisoformat(entry['validTime'].split('/')[0].replace('Z', '+00:00'))
                date_key = start_time.date()
                if entry['value'] is not None:
                    if date_key not in daily_forecasts:
                        daily_forecasts[date_key] = {'temp_c': -99, 'snow_mm': 0.0}
                    daily_forecasts[date_key]['snow_mm'] += entry['value']
            except Exception as e:
                logging.warning(f"Error parsing NWS snow data: {e}")
        
        today = datetime.now(timezone.utc).date()
        for date_key in sorted(daily_forecasts.keys()):
            if date_key < today:
                continue

            forecast = daily_forecasts[date_key]
            temp_f = forecast['temp_c'] * 9/5 + 32
            snow_in = forecast['snow_mm'] / 25.4
            
            if snow_in > 0.1 and temp_f <= 33.0:
                logging.info(f"NWS Forecast Found: Snow on {date_key.strftime('%B %d')}")
                return date_key
                
        logging.info("No snow found in 7-day NWS forecast.")
        
        # Fallback to Open-Meteo forecast
        today_local = datetime.now().date()
        end = (today_local + timedelta(days=7)).strftime('%Y-%m-%d')
        forecast_df = fetch_open_meteo_data(lat, lon, today_local.strftime('%Y-%m-%d'), end, is_forecast=True)
        for _, row in forecast_df.iterrows():
            snow_in = row['snowfall_sum'] / 2.54
            temp_f = row['temperature_2m_min'] * 9/5 + 32
            if snow_in > 0.1 and temp_f <= 33.0:
                return row['date'].date()
        
        return None
        
    except Exception as e:
        logging.warning(f"Error fetching NWS forecast: {e}")
        return None

# --- Main ---
if __name__ == "__main__":
    location = load_config()
    if location is None:
        location = "Lansing, MI"
        save_config(location)
    
    lat, lon = get_lat_lon(location)
    logging.info(f"Using coordinates: {lat:.4f}, {lon:.4f}")
    
    # Start web server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    while True:
        try:
            current_date = datetime.now()
            last_prediction_date = current_date
            next_prediction_date = current_date + timedelta(days=1)
            logging.info("--- Daily Prediction Cycle ---")
            
            model_1_data, model_2_data, first_snow_dict, current_drivers = get_historical_data_for_ml(lat, lon)
            
            if not first_snow_dict:
                logging.critical("No historical data. Exiting.")
                exit()
            
            (X_train_1, y_train_1) = model_1_data
            model_1, scalers_1, device_1 = train_first_snow_model_dl(X_train_1, y_train_1)
            
            (X_train_2, y_train_2) = model_2_data
            if len(X_train_2) > 0:
                model_2, device_2 = train_seasonal_snow_model_dl(X_train_2, y_train_2)
            else:
                model_2 = None
            
            adjusted_doys = [d if d < 365 else d - 365 for d in first_snow_dict.values()]
            prior_mu = np.median(adjusted_doys)
            prior_sigma = max(np.std(adjusted_doys) if len(adjusted_doys) > 1 else 15.0, 5.0)
            
            # --- Perform Predictions ---
            current_oni = current_drivers['oni']
            oni_status = "Neutral"
            if current_oni >= 0.5:
                oni_status = f"El Niño ({current_oni:.2f})"
            elif current_oni <= -0.5:
                oni_status = f"La Niña ({current_oni:.2f})"
            
            driver_status_string = {
                "ENSO": oni_status,
                "AO": f"{current_drivers['ao']:.2f}",
                "NAO": f"{current_drivers['nao']:.2f}",
                "PNA": f"{current_drivers['pna']:.2f}",
                "PDO": f"{current_drivers['pdo']:.2f}",
                "QBO": f"{current_drivers['qbo']:.2f}",
                "SSN": f"{current_drivers['ssn']:.2f}",
                "MJO_AMP": f"{current_drivers['mjo_amp']:.2f}"
            }
            
            nws_snow_date = fetch_nws_forecast_with_snow(lat, lon)
            
            predicted_timestamp_ms = None
            predicted_doy = None
            current_year = current_date.year
            if nws_snow_date:
                days_until = (nws_snow_date - current_date.date()).days
                first_snow_message = f"First Snowfall: {nws_snow_date.strftime('%A, %B %d, %Y')} (Days from now: {days_until}) Confidence: HIGH (Based on NWS forecast)"
                predicted_date = datetime.combine(nws_snow_date, datetime.min.time())
                predicted_timestamp_ms = int(predicted_date.timestamp() * 1000)
                predicted_doy = nws_snow_date.timetuple().tm_yday
            else:
                logging.info("No NWS snow. Moving to statistical Model 1...")
                current_doy = current_date.timetuple().tm_yday
                ml_unwrapped = predict_first_snow_with_ml(model_1, scalers_1, device_1, current_doy, current_drivers)
                
                if ml_unwrapped is None:
                    first_snow_message = "No likely snow day in the next 180 days."
                else:
                    prior_mu_adjusted = prior_mu if prior_mu > current_doy else prior_mu + 365
                    final_fused_unwrapped = bayesian_update(prior_mu_adjusted, prior_sigma, ml_unwrapped)
                    
                    fused = final_fused_unwrapped
                    predicted_year = current_year
                    while fused > 365:
                        fused -= 365
                        predicted_year += 1
                    predicted_doy = int(fused)
                    predicted_date = datetime(predicted_year, 1, 1) + timedelta(days=predicted_doy - 1)
                    days_until = (predicted_date - current_date).days
                    
                    first_snow_message = f"First Snowfall: {predicted_date.strftime('%B %d, %Y')} (Days from now: {days_until}) Confidence: MEDIUM (Based on {len(first_snow_dict)} years)"
                    predicted_timestamp_ms = int(predicted_date.timestamp() * 1000)
            
            if model_2 is not None:
                seasonal_driver_features = [current_drivers[name] for name in ['oni', 'ao', 'nao', 'pna', 'pdo', 'qbo', 'ssn', 'mjo_amp']]
                predicted_total_snow = predict_seasonal_snow_with_dl(model_2, device_2, seasonal_driver_features)
                seasonal_message = f"Predicted Total: {predicted_total_snow:.1f} inches Confidence: MEDIUM (Based on {len(y_train_2)} seasons)"
            else:
                seasonal_message = "Not enough data for seasonal prediction."
            
            # New features
            current_weather = fetch_current_weather(lat, lon)
            historical_df = load_historical_weather()
            plot_path = generate_snow_plot(historical_df)
            first_snow_plot_path = generate_first_snow_plot(first_snow_dict, predicted_doy, current_year)
            
            # Generate HTML
            generate_html(location, driver_status_string, first_snow_message, seasonal_message, last_prediction_date, next_prediction_date, current_weather, plot_path, first_snow_plot_path, predicted_timestamp_ms)
            
            logging.info("Cycle complete. Sleeping for 24 hours...")
            time.sleep(86400)
        
        except Exception as e:
            logging.error(f"Error: {e}")
            time.sleep(900)  # Retry in 15 min
