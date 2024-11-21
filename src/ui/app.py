import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import yaml
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Import local modules using absolute imports
from src.utils.data_processing import preprocess_df, encoding_categorical_features

# Load config
config_path = PROJECT_ROOT / 'src' / 'config' / 'config.yaml'
with open(config_path) as infile:
    config = yaml.safe_load(infile)
features_to_use = config["features_to_use"]
columns_to_scale = config["columns_to_scale"]

# Page config
st.set_page_config(page_title="Marketing Mix Model Optimizer", layout="wide")

# Load the model and data
@st.cache_resource
def load_model():
    """Uses pickle to load trained model. 

    Returns:
        sklearn.linear_model._base.LinearRegression : Trained MMM model
    """
    model_path = PROJECT_ROOT / 'src' / 'models' / 'model.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    """Loading the sample_media_spend_data csv as df,
       performing basic preprocessing.

    Returns:
        pandas.core.frame.DataFrame: loaded dataset as df
    """
    data_path = PROJECT_ROOT / 'raw_data' / 'sample_media_spend_data.csv'
    df = pd.read_csv(data_path)
    df = df.reset_index(drop=True)
    df.columns = map(str.lower, df.columns)
    return df

model = load_model()
df = load_data()
scaler = MinMaxScaler()

# Title
st.title("Marketing Mix Model Optimization Dashboard")

# Sidebar for global controls
st.sidebar.header("Controls")
selected_division = st.sidebar.selectbox("Select Division", sorted(df['division'].unique()))

# Filter data for selected division
df_filtered = df[df['division'] == selected_division].copy()

# ROI Analysis 
st.header("Channel ROI Analysis")

# Calculate channel ROI
channels = ['paid_views', 'google_impressions', 'email_impressions', 
            'facebook_impressions', 'affiliate_impressions']

roi_data = []
for channel in channels:
    avg_spend = df_filtered[channel].mean()
    
    # Create two scenarios for ROI calculation
    base_scenario = df_filtered.copy()
    increased_scenario = df_filtered.copy()
    increased_scenario[channel] = increased_scenario[channel] * 1.1  # 10% increase
    
    # Prepare data for prediction
    # Data preprocessing and encoding
    base_scenario = preprocess_df(base_scenario)
    increased_scenario = preprocess_df(increased_scenario)
    base_scenario = encoding_categorical_features(base_scenario, columns_to_scale)
    increased_scenario = encoding_categorical_features(increased_scenario, columns_to_scale)

    # Scale data
    base_scenario[columns_to_scale] = scaler.fit_transform(base_scenario[columns_to_scale])
    increased_scenario[columns_to_scale] = scaler.transform(increased_scenario[columns_to_scale])
    
    # Get predictions
    base_pred = model.predict(base_scenario[features_to_use])
    increased_pred = model.predict(increased_scenario[features_to_use])
    
    # Calculate ROI
    sales_lift = (increased_pred - base_pred).mean()
    # Simulating a 10% increase in investment 
    investment_increase = avg_spend * 0.1
    # ROI computed as change in sales over change in investment
    roi = (sales_lift / investment_increase) if investment_increase > 0 else 0
    
    roi_data.append({
        'Channel': channel.replace('_', ' ').title(),
        'ROI': roi,
        'Avg Spend': avg_spend,
        'Sales Lift per 10% Increase': sales_lift
    })

roi_df = pd.DataFrame(roi_data)

# ROI visualization
fig = px.bar(roi_df, x='Channel', y='ROI', 
                title='Return on Investment by Channel',
                color='ROI')
st.plotly_chart(fig)

# Detailed metrics
st.dataframe(roi_df)
