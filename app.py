import streamlit as st
import pandas as pd
import plotly.express as px
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from predict import load_model_and_preprocessor, predict

st.set_page_config(page_title="Bike Sharing Predictor", layout="wide")

# Load model and preprocessor
try:
    model, preprocessor = load_model_and_preprocessor()
    st.success("Model and preprocessor loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model/preprocessor: {str(e)}")
    st.stop()

st.title("🚴‍♂️ Real-Time Bike Sharing Pattern Prediction")

tab1, tab2 = st.tabs(["Manual Prediction", "Real-Time Simulation"])

# Manual Prediction Tab (Unchanged)
with tab1:
    st.header("Manual Prediction")
    st.write("Enter parameters to predict bike-sharing demand instantly.")

    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperature (°C)", -10, 40, 20, help="Temperature in Celsius")
        humidity = st.slider("Humidity (%)", 0, 100, 50, help="Relative humidity")
        windspeed = st.slider("Wind Speed (km/h)", 0, 50, 10, help="Wind speed")
        hour = st.slider("Hour (0-23)", 0, 23, 8, help="Hour of the day (24-hour format)")
        day = st.slider("Day (1-31)", 1, 31, 1, help="Day of the month")
        month = st.slider("Month (1-12)", 1, 12, 6, help="Month of the year")
    
    with col2:
        holiday = st.selectbox("Holiday", [0, 1], index=0, help="0 = No, 1 = Yes")
        workingday = st.selectbox("Working Day", [0, 1], index=1, help="0 = No, 1 = Yes")
        dayofweek = st.selectbox("Day of Week", list(range(7)), index=2, help="0 = Monday, 6 = Sunday")
        weather = st.selectbox("Weather", [0, 1, 2, 3], index=0,
                              format_func=lambda x: ["Clear", "Cloudy", "Rain", "Snow"][x])
        season = st.selectbox("Season", [1, 2, 3, 4], index=0,
                              format_func=lambda x: ["Spring", "Summer", "Fall", "Winter"][x - 1])
    
    input_data = {
        'temp': temp,
        'humidity': humidity,
        'windspeed': windspeed,
        'hour': hour,
        'day': day,
        'month': month,
        'holiday': holiday,
        'workingday': workingday,
        'dayofweek': dayofweek,
        'weather': weather,
        'season': season
    }
    
    if st.button("Predict", key="manual_predict"):
        try:
            pred, clas = predict(model, preprocessor, input_data)
            
            st.subheader("Prediction Results")
            col_result1, col_result2 = st.columns([1, 2])
            with col_result1:
                st.metric("Predicted Demand", f"{pred:.2f} bikes")
                st.write(f"Classification: **{clas}**")
            
            with col_result2:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie([pred, 200 - pred] if pred <= 200 else [200, pred - 200], 
                       startangle=90, colors=['#FF6F61' if clas == "High" else '#6BBF59', '#E0E0E0'],
                       wedgeprops={'width': 0.3})
                ax.text(0, 0, f"{pred:.0f}", ha='center', va='center', fontsize=20)
                ax.set_title("Demand Gauge (Max 200)")
                st.pyplot(fig)
            
            with st.expander("View Input Parameters"):
                st.json(input_data)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Real-Time Simulation Tab (Updated to Remove Static Plot)
with tab2:
    st.header("Real-Time Simulation")
    st.write("Simulate bike-sharing demand over time with automatic updates every 2 seconds. Click 'Run Simulation' to start.")

    # Initialize session state variables
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()
    if 'current_data' not in st.session_state:
        # Initialize with a starting data point
        st.session_state.current_data = {
            'datetime': datetime.now(),
            'season': 1,
            'holiday': 0,
            'workingday': 0,
            'weather': 1,
            'temp': 20.0,
            'humidity': 50.0,
            'windspeed': 10.0,
            'hour': 0,
            'day': 1,
            'month': 1,
            'dayofweek': 0
        }

    # Button to start/stop the simulation
    if st.button("Run Simulation", key="run_simulation"):
        st.session_state.simulation_running = not st.session_state.simulation_running
        if st.session_state.simulation_running:
            st.write("Simulation started. Updates every 2 seconds...")
        else:
            st.write("Simulation stopped.")

    # Simulation logic
    if st.session_state.simulation_running:
        current_time = time.time()
        if current_time - st.session_state.last_update >= 2:  # Update every 2 seconds
            # Generate new data point based on the previous one
            new_data = st.session_state.current_data.copy()
            new_data['datetime'] += pd.Timedelta(hours=1)  # Increment time by 1 hour
            new_data['hour'] = (new_data['hour'] + 1) % 24
            if new_data['hour'] == 0:
                new_data['day'] = (new_data['day'] % 31) + 1
                if new_data['day'] == 1:
                    new_data['month'] = (new_data['month'] % 12) + 1
            new_data['dayofweek'] = new_data['datetime'].weekday()

            # Simulate realistic changes in features
            new_data['temp'] += np.random.uniform(-1, 1)
            new_data['humidity'] += np.random.uniform(-5, 5)
            new_data['windspeed'] += np.random.uniform(-2, 2)
            new_data['temp'] = np.clip(new_data['temp'], -10, 40)
            new_data['humidity'] = np.clip(new_data['humidity'], 0, 100)
            new_data['windspeed'] = np.clip(new_data['windspeed'], 0, 50)
            if np.random.rand() < 0.1:  # 10% chance to change weather
                new_data['weather'] = np.random.choice([1, 2, 3, 4])

            st.session_state.current_data = new_data

            # Prepare input for prediction
            input_data = {
                'temp': new_data['temp'],
                'humidity': new_data['humidity'],
                'windspeed': new_data['windspeed'],
                'hour': new_data['hour'],
                'day': new_data['day'],
                'month': new_data['month'],
                'holiday': new_data['holiday'],
                'workingday': new_data['workingday'],
                'dayofweek': new_data['dayofweek'],
                'weather': new_data['weather'],
                'season': new_data['season']
            }

            # Make prediction
            try:
                pred, clas = predict(model, preprocessor, input_data)
                st.session_state.history.append({
                    'time': new_data['datetime'],
                    'prediction': pred,
                    'classification': clas
                })
                # Limit history to last 100 points
                if len(st.session_state.history) > 100:
                    st.session_state.history.pop(0)
                st.session_state.last_update = current_time
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

    # Display latest prediction and dynamic plot
    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.subheader("Latest Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Time", latest['time'].strftime("%Y-%m-%d %H:%M:%S"))
            st.metric("Predicted Demand", f"{latest['prediction']:.2f} bikes")
        with col2:
            st.write(f"Classification: **{latest['classification']}**")

        # Dynamic plot using Plotly (moving window) with a placeholder
        df_history = pd.DataFrame(st.session_state.history)
        WINDOW_SIZE = 20  # Define window size for moving plot
        start_idx = max(0, len(df_history) - WINDOW_SIZE)
        window_data = df_history[start_idx:]
        
        fig = px.line(window_data, x='time', y='prediction', title="Bike Demand Over Time (Moving Window)",
                      labels={'time': 'Time', 'prediction': 'Predicted Demand (Bikes)'})
        fig.add_scatter(x=[latest['time']], y=[latest['prediction']],
                        mode='markers',
                        marker=dict(color='red' if latest['classification'] == "High" else 'green', size=10),
                        name='Latest Prediction')
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Predicted Demand (Bikes)",
            xaxis=dict(tickangle=45, range=[min(window_data['time']), max(window_data['time'])]),
            yaxis=dict(autorange=True),
            showlegend=False
        )

        # Use a placeholder to update the plot in place
        if 'plot_placeholder' not in st.session_state:
            st.session_state.plot_placeholder = st.empty()  # Create a placeholder once
        st.session_state.plot_placeholder.plotly_chart(fig, use_container_width=True)  # Update the placeholder
    else:
        st.write("No predictions yet. Click 'Run Simulation' to start live updates.")

    # Rerun the app periodically if simulation is running
    if st.session_state.simulation_running:
        time.sleep(0.1)  # Small delay to prevent excessive reruns
        st.rerun()