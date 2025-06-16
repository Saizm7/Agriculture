import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="Grape Farmer Assistant", layout="wide")

# Indian grape varieties
GRAPE_VARIETIES = {
    'Table Grapes': {
        'Thompson Seedless': {
            'description': 'Most popular variety in India, known for its sweetness and seedlessness',
            'growing_regions': ['Nashik', 'Sangli', 'Solapur'],
            'harvest_season': 'April-June',
            'market_price_range': '₹85-125/kg'
        },
        'Flame Seedless': {
            'description': 'Red seedless variety with good shelf life',
            'growing_regions': ['Nashik', 'Pune'],
            'harvest_season': 'May-July',
            'market_price_range': '₹90-130/kg'
        },
        'Manjari Naveen': {
            'description': 'Early maturing variety with good yield',
            'growing_regions': ['Nashik', 'Sangli'],
            'harvest_season': 'March-May',
            'market_price_range': '₹80-120/kg'
        }
    },
    'Wine Grapes': {
        'Cabernet Sauvignon': {
            'description': 'Premium wine grape variety',
            'growing_regions': ['Nashik', 'Sangli'],
            'harvest_season': 'February-April',
            'market_price_range': '₹65-95/kg'
        },
        'Shiraz': {
            'description': 'Popular for red wine production',
            'growing_regions': ['Nashik', 'Sangli'],
            'harvest_season': 'February-April',
            'market_price_range': '₹60-90/kg'
        }
    },
    'Raisin Grapes': {
        'Sonaka': {
            'description': 'Special variety for raisin production',
            'growing_regions': ['Sangli', 'Solapur'],
            'harvest_season': 'March-May',
            'market_price_range': '₹75-105/kg'
        }
    }
}

# Disease information with more accurate details
DISEASE_INFO = {
    'healthy': {
        'name': 'Healthy Grape',
        'description': 'The grape plant shows normal growth with healthy green leaves and proper fruit development.',
        'treatment': 'No treatment needed. Continue regular maintenance practices.',
        'prevention': 'Maintain proper spacing (2-3m), regular pruning, balanced fertilization (NPK 19:19:19), and proper irrigation.'
    },
    'leaf_blight': {
        'name': 'Leaf Blight (Isariopsis Leaf Spot)',
        'description': 'A fungal disease common in Indian vineyards, especially during monsoon season (June-September). Characterized by brown spots with yellow halos on leaves.',
        'treatment': 'Apply fungicides: Mancozeb (2g/l) or Copper oxychloride (3g/l) at 15-day intervals. Remove infected leaves.',
        'prevention': 'Ensure proper drainage, maintain plant spacing, avoid overhead irrigation, and remove fallen leaves.'
    },
    'esca': {
        'name': 'Esca (Black Measles)',
        'description': 'A serious fungal disease affecting older grapevines in India. Shows as dark spots on leaves and berries, with internal wood decay.',
        'treatment': 'Prune affected parts and apply systemic fungicides like Tebuconazole (1ml/l). Remove and destroy infected plants.',
        'prevention': 'Use disease-free planting material, maintain proper vineyard hygiene, and avoid wounding during pruning.'
    },
    'black_rot': {
        'name': 'Black Rot',
        'description': 'A fungal disease prevalent in humid Indian conditions. Causes circular brown spots on leaves and black, shriveled berries.',
        'treatment': 'Apply fungicides: Mancozeb (2g/l) or Carbendazim (1g/l) at 10-day intervals. Remove infected berries.',
        'prevention': 'Remove infected berries, maintain proper canopy management, ensure good air circulation, and avoid overhead irrigation.'
    }
}

# India-specific cultivation calendar
CULTIVATION_CALENDAR = {
    'January': [
        'Winter pruning (15-30 January)',
        'Apply basal fertilizers (NPK 19:19:19 @ 500g/vine)',
        'Start drip irrigation (2-3 days interval)',
        'Apply Bordeaux mixture for disease prevention'
    ],
    'February': [
        'Bud break monitoring',
        'Apply growth regulators (GA3 @ 10ppm)',
        'Monitor for mealybug and thrips',
        'Start shoot training'
    ],
    'March': [
        'Flowering stage management',
        'Apply micronutrients (Zn, B, Ca)',
        'Control weeds (manual + chemical)',
        'Maintain proper canopy density'
    ],
    'April': [
        'Fruit set management',
        'Cluster thinning (8-10 clusters/vine)',
        'Apply fungicides for disease control',
        'Irrigation scheduling (3-4 days interval)'
    ],
    'May': [
        'Berry development stage',
        'Irrigation management (2-3 days interval)',
        'Pest control (especially mealybug)',
        'Apply potassium-rich fertilizers'
    ],
    'June': [
        'Harvesting (15-30 June)',
        'Post-harvest care',
        'Pruning for next season',
        'Apply organic manure'
    ],
    'July': [
        'Vine training and trellising',
        'Soil preparation and testing',
        'Apply organic manure (10kg/vine)',
        'Start irrigation for new growth'
    ],
    'August': [
        'Canopy management',
        'Disease monitoring',
        'Irrigation scheduling',
        'Apply balanced fertilizers'
    ],
    'September': [
        'Fertilizer application (NPK 19:19:19)',
        'Pest monitoring and control',
        'Weed control',
        'Maintain proper vine spacing'
    ],
    'October': [
        'Leaf removal for better air circulation',
        'Cluster thinning',
        'Disease prevention sprays',
        'Irrigation management'
    ],
    'November': [
        'Frost protection measures',
        'Pruning preparation',
        'Soil testing and amendment',
        'Apply organic mulch'
    ],
    'December': [
        'Winter pruning preparation',
        'Dormant spraying',
        'Equipment maintenance',
        'Plan next season activities'
    ]
}

# Real market prices for Indian markets (updated quarterly)
MARKET_PRICES = {
    'Table Grapes (Nashik)': {'min': 85, 'max': 125, 'unit': 'kg'},
    'Table Grapes (Sangli)': {'min': 80, 'max': 120, 'unit': 'kg'},
    'Table Grapes (Solapur)': {'min': 75, 'max': 115, 'unit': 'kg'},
    'Wine Grapes (Nashik)': {'min': 65, 'max': 95, 'unit': 'kg'},
    'Wine Grapes (Sangli)': {'min': 60, 'max': 90, 'unit': 'kg'},
    'Raisin Grapes (Sangli)': {'min': 75, 'max': 105, 'unit': 'kg'},
    'Raisin Grapes (Solapur)': {'min': 70, 'max': 100, 'unit': 'kg'}
}

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

# Get weather data from OpenWeatherMap API
@st.cache_data(ttl=3600)
def get_weather_data(location="Nashik, Maharashtra"):
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    # Location coordinates for major grape-growing regions in Maharashtra
    LOCATION_COORDS = {
        "Nashik, Maharashtra": {"lat": 20.0059, "lon": 73.7914},
        "Sangli, Maharashtra": {"lat": 16.8524, "lon": 74.5815},
        "Solapur, Maharashtra": {"lat": 17.6599, "lon": 75.9064},
        "Pune, Maharashtra": {"lat": 18.5204, "lon": 73.8567}
    }
    
    # Check if API key is set and valid
    if not api_key:
        st.error("OpenWeather API key not found in .env file. Please check your configuration.")
        return get_mock_weather_data()
    
    # Remove any whitespace from API key
    api_key = api_key.strip()
    
    try:
        # Get coordinates for the selected location
        coords = LOCATION_COORDS.get(location)
        if not coords:
            st.warning(f"Location {location} not found in coordinates database. Using mock data.")
            return get_mock_weather_data()
        
        # Get current weather using lat/lon
        current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={coords['lat']}&lon={coords['lon']}&appid={api_key}&units=metric"
        current_response = requests.get(current_url)
        
        if current_response.status_code == 401:
            st.error("Invalid API key. Please check your OpenWeather API key in the .env file.")
            st.info("Current API key: " + api_key[:8] + "..." + api_key[-4:])
            return get_mock_weather_data()
        elif current_response.status_code != 200:
            st.warning(f"Could not fetch weather data for {location}. Status code: {current_response.status_code}")
            return get_mock_weather_data()
            
        current_data = current_response.json()
        
        # Get forecast using lat/lon
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={coords['lat']}&lon={coords['lon']}&appid={api_key}&units=metric"
        forecast_response = requests.get(forecast_url)
        
        if forecast_response.status_code != 200:
            st.warning(f"Could not fetch forecast data for {location}. Status code: {forecast_response.status_code}")
            return get_mock_weather_data()
            
        forecast_data = forecast_response.json()
        
        # Process current weather
        current_weather = {
            'temperature': round(current_data.get('main', {}).get('temp', 28)),
            'humidity': current_data.get('main', {}).get('humidity', 65),
            'rainfall': current_data.get('rain', {}).get('1h', 0),
            'condition': current_data.get('weather', [{}])[0].get('main', 'Sunny')
        }
        
        # Process forecast
        forecast = []
        for i in range(1, 4):
            try:
                day_data = forecast_data.get('list', [{}])[i*8]  # Get data for each day at noon
                forecast.append({
                    'date': datetime.fromtimestamp(day_data.get('dt', 0)).strftime('%A'),
                    'temp': round(day_data.get('main', {}).get('temp', 28)),
                    'condition': day_data.get('weather', [{}])[0].get('main', 'Sunny')
                })
            except (IndexError, KeyError, TypeError):
                forecast.append({
                    'date': f'Day {i}',
                    'temp': 28,
                    'condition': 'Sunny'
                })
        
        return {
            'temperature': current_weather['temperature'],
            'humidity': current_weather['humidity'],
            'rainfall': current_weather['rainfall'],
            'forecast': forecast
        }
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return get_mock_weather_data()

def get_mock_weather_data():
    """Return mock weather data when API fails or is not available"""
    return {
        'temperature': 28,
        'humidity': 65,
        'rainfall': 0,
        'forecast': [
            {'date': 'Today', 'temp': 28, 'condition': 'Sunny'},
            {'date': 'Tomorrow', 'temp': 26, 'condition': 'Partly Cloudy'},
            {'date': 'Day 3', 'temp': 25, 'condition': 'Light Rain'}
        ]
    }

# Main app
st.title("🍇 Grape Farmer Assistant")
st.write("Your comprehensive guide for grape cultivation and disease management in India")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Choose a feature",
    ["Disease Detection", "Cultivation Calendar", "Weather Info", "Market Prices", "Expert Tips", "Grape Varieties"]
)

if page == "Disease Detection":
    st.header("Disease Detection")
    st.write("Upload an image of a grape leaf to detect diseases.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display and process image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process image
        image = image.convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        model = load_model()
        prediction = model.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction[0])
        classes = ['healthy', 'leaf_blight', 'esca', 'black_rot']
        predicted_class = classes[class_idx]
        confidence = float(prediction[0][class_idx])
        
        # Display results
        st.header("Results")
        st.write(f"**Condition:** {DISEASE_INFO[predicted_class]['name']}")
        st.write(f"**Confidence:** {confidence:.2%}")
        st.write(f"**Treatment:** {DISEASE_INFO[predicted_class]['treatment']}")
        
        # Display disease information
        st.subheader("Disease Information")
        st.write(f"**Description:** {DISEASE_INFO[predicted_class]['description']}")
        st.write(f"**Prevention:** {DISEASE_INFO[predicted_class]['prevention']}")

elif page == "Cultivation Calendar":
    st.header("Cultivation Calendar")
    current_month = datetime.now().strftime("%B")
    
    # Display current month's tasks
    st.subheader(f"Tasks for {current_month}")
    for task in CULTIVATION_CALENDAR[current_month]:
        st.write(f"✓ {task}")
    
    # Display full calendar
    st.subheader("Full Year Calendar")
    calendar_df = pd.DataFrame.from_dict(CULTIVATION_CALENDAR, orient='index')
    st.dataframe(calendar_df, use_container_width=True)

elif page == "Weather Info":
    st.header("Weather Information")
    
    # Location selection
    location = st.selectbox(
        "Select Location",
        ["Nashik, Maharashtra", "Sangli, Maharashtra", "Solapur, Maharashtra", "Pune, Maharashtra"]
    )
    
    weather_data = get_weather_data(location)
    
    # Current weather
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Temperature", f"{weather_data['temperature']}°C")
    with col2:
        st.metric("Humidity", f"{weather_data['humidity']}%")
    with col3:
        st.metric("Rainfall", f"{weather_data['rainfall']} mm")
    
    # Weather forecast
    st.subheader("3-Day Forecast")
    forecast_df = pd.DataFrame(weather_data['forecast'])
    st.dataframe(forecast_df, use_container_width=True)
    
    # Weather alerts
    if weather_data['temperature'] > 35:
        st.warning("High temperature alert! Ensure proper irrigation and canopy management.")
    if weather_data['humidity'] > 80:
        st.warning("High humidity alert! Monitor for fungal diseases and ensure proper ventilation.")
    if weather_data['rainfall'] > 0:
        st.info("Rainfall detected. Consider applying preventive fungicides.")

elif page == "Market Prices":
    st.header("Market Prices")
    
    # Display current prices
    for grape_type, price_info in MARKET_PRICES.items():
        st.subheader(grape_type)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Minimum Price", f"₹{price_info['min']}/{price_info['unit']}")
        with col2:
            st.metric("Maximum Price", f"₹{price_info['max']}/{price_info['unit']}")
    
    # Price trends (mock data)
    st.subheader("Price Trends")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
    prices = np.random.normal(100, 10, len(dates))
    df = pd.DataFrame({'Date': dates, 'Price': prices})
    fig = px.line(df, x='Date', y='Price', title='Grape Price Trends')
    st.plotly_chart(fig)

elif page == "Expert Tips":
    st.header("Expert Tips")
    
    # Cultivation tips
    st.subheader("Cultivation Tips")
    tips = [
        "Plant grapes in well-drained soil with pH between 5.5 and 7.0",
        "Maintain proper spacing (2-3 meters between vines)",
        "Use drip irrigation for efficient water management",
        "Prune vines during dormancy (December-January)",
        "Apply balanced fertilizers (NPK 19:19:19) during growing season",
        "Monitor soil moisture regularly",
        "Use organic mulch to retain soil moisture",
        "Implement proper trellising system",
        "Control weeds regularly",
        "Harvest grapes at proper maturity"
    ]
    
    for tip in tips:
        st.write(f"• {tip}")
    
    # Common mistakes to avoid
    st.subheader("Common Mistakes to Avoid")
    mistakes = [
        "Over-irrigation leading to root rot",
        "Improper pruning causing reduced yield",
        "Late disease detection and treatment",
        "Incorrect fertilizer application timing",
        "Poor canopy management",
        "Neglecting pest control",
        "Harvesting immature grapes",
        "Improper storage conditions"
    ]
    
    for mistake in mistakes:
        st.write(f"⚠️ {mistake}")

elif page == "Grape Varieties":
    st.header("Indian Grape Varieties")
    
    # Display varieties by category
    for category, varieties in GRAPE_VARIETIES.items():
        st.subheader(category)
        for variety, info in varieties.items():
            with st.expander(variety):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Growing Regions:** {', '.join(info['growing_regions'])}")
                st.write(f"**Harvest Season:** {info['harvest_season']}")
                st.write(f"**Market Price Range:** {info['market_price_range']}") 