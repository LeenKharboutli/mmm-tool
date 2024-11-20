import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from data_processing import *

with open('config.yaml') as infile:
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
    return pickle.load(open('analysis/model.pkl', 'rb'))

@st.cache_data
def load_data():
    """Loading the sample_media_spend_data csv as df,
       performing basic preprocessing.

    Returns:
        pandas.core.frame.DataFrame: loaded dataset as df
    """
    df = pd.read_csv('raw_data/sample_media_spend_data.csv')
    df = df.reset_index(drop = True)
    df.columns = map(str.lower, df.columns)
    return df

model = load_model()
df = load_data()

# NOTE: I think we have to rescale with each modification! 
# NOTE: does that screw with our model? 


# Title
st.title("Marketing Mix Model Optimization Dashboard")

# Sidebar for global controls
st.sidebar.header("Controls")
selected_division = st.sidebar.selectbox("Select Division", sorted(df['division'].unique()))

# Filter data for selected division
df_filtered = df[df['division'] == selected_division].copy()

# Main dashboard sections
# tab1, tab2, tab3 = st.tabs(["ROI Analysis", "Budget Optimizer", "Scenario Planning"])

# NOTE: preprocessing: not sure exactly where to put this...
df_filtered = preprocess_df(df_filtered)
df_filtered = encoding_categorical_features(df_filtered, columns_to_scale)


# ROI Analysis Tab
# with tab1:
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
    # NOTE: would this be done again? 
    # scaler = MinMaxScaler()
    # features_scaled = scaler.fit_transform(base_scenario[channels])
    # features_scaled_increased = scaler.transform(increased_scenario[channels])
    
    # Get predictions
    base_pred = model.predict(base_scenario[features_to_use])
    increased_pred = model.predict(increased_scenario[features_to_use])
    
    # Calculate ROI
    sales_lift = (increased_pred - base_pred).mean()
    investment_increase = avg_spend * 0.1
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

# # Budget Optimizer Tab
# with tab2:
#     st.header("Interactive Budget Allocation Tool")
    
#     # Current budget allocation
#     current_allocation = {channel: df_filtered[channel].mean() 
#                          for channel in channels}
    
#     # Budget sliders
#     st.subheader("Adjust Channel Budgets")
#     total_budget = sum(current_allocation.values())
    
#     new_allocation = {}
#     col1, col2 = st.columns(2)
    
#     with col1:
#         for channel in channels[:3]:
#             channel_name = channel.replace('_', ' ').title()
#             new_allocation[channel] = st.slider(
#                 f"{channel_name} Budget",
#                 min_value=0.0,
#                 max_value=total_budget,
#                 value=float(current_allocation[channel]),
#                 format="%.2f"
#             )
    
#     with col2:
#         for channel in channels[3:]:
#             channel_name = channel.replace('_', ' ').title()
#             new_allocation[channel] = st.slider(
#                 f"{channel_name} Budget",
#                 min_value=0.0,
#                 max_value=total_budget,
#                 value=float(current_allocation[channel]),
#                 format="%.2f"
#             )
    
#     # Predict sales with new allocation
#     new_data = df_filtered.copy()
#     for channel in channels:
#         new_data[channel] = new_allocation[channel]
    
#     # scaler = MinMaxScaler()
#     # features_scaled = scaler.fit_transform(new_data[channels])
#     predicted_sales = model.predict(new_data[features_to_use])
    
#     # Display predictions
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Predicted Sales", f"${predicted_sales.mean():,.2f}")
#     with col2:
#         current_sales = model.predict(df_filtered[features_to_use])
#         sales_diff = predicted_sales.mean() - current_sales.mean()
#         st.metric("Sales Impact", f"${sales_diff:,.2f}", 
#                  delta=f"{(sales_diff/current_sales.mean())*100:.1f}%")

# # Scenario Planning Tab
# with tab3:
#     st.header("What-If Scenario Planning")
    
#     # Create scenarios
#     st.subheader("Create and Compare Scenarios")
    
#     # Scenario inputs
#     col1, col2 = st.columns(2)
    
#     scenarios = {}
    
#     with col1:
#         st.write("Scenario A")
#         scenarios['A'] = {}
#         for channel in channels:
#             channel_name = channel.replace('_', ' ').title()
#             scenarios['A'][channel] = st.number_input(
#                 f"Scenario A - {channel_name}",
#                 min_value=0.0,
#                 value=float(df_filtered[channel].mean()),
#                 key=f"A_{channel}"
#             )
    
#     with col2:
#         st.write("Scenario B")
#         scenarios['B'] = {}
#         for channel in channels:
#             channel_name = channel.replace('_', ' ').title()
#             scenarios['B'][channel] = st.number_input(
#                 f"Scenario B - {channel_name}",
#                 min_value=0.0,
#                 value=float(df_filtered[channel].mean()),
#                 key=f"B_{channel}"
#             )
    
#     # Compare scenarios
#     if st.button("Compare Scenarios"):
#         results = []
        
#         for scenario in ['A', 'B']:
#             # Prepare data for prediction
#             scenario_data = df_filtered.copy()
#             for channel in channels:
#                 scenario_data[channel] = scenarios[scenario][channel]
            
#             # NOTE: not sure yet if I should scale first!
#             # Scale features and predict
#             scaler = MinMaxScaler()
#             features_scaled = scaler.fit_transform(scenario_data[channels])
#             predicted_sales = model.predict(scenario_data[features_to_use])
            
#             results.append({
#                 'Scenario': scenario,
#                 'Predicted Sales': predicted_sales.mean(),
#                 'Total Investment': sum(scenarios[scenario].values())
#             })
        
#         results_df = pd.DataFrame(results)
        
#         # Display comparison
#         st.subheader("Scenario Comparison")
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.metric("Scenario A Sales", f"${results_df.iloc[0]['Predicted Sales']:,.2f}")
#             st.metric("Scenario A Investment", f"${results_df.iloc[0]['Total Investment']:,.2f}")
        
#         with col2:
#             st.metric("Scenario B Sales", f"${results_df.iloc[1]['Predicted Sales']:,.2f}")
#             st.metric("Scenario B Investment", f"${results_df.iloc[1]['Total Investment']:,.2f}")
        
#         # ROI comparison
#         roi_a = (results_df.iloc[0]['Predicted Sales'] / results_df.iloc[0]['Total Investment']) if results_df.iloc[0]['Total Investment'] > 0 else 0
#         roi_b = (results_df.iloc[1]['Predicted Sales'] / results_df.iloc[1]['Total Investment']) if results_df.iloc[1]['Total Investment'] > 0 else 0
        
#         st.subheader("ROI Comparison")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Scenario A ROI", f"{roi_a:.2f}")
#         with col2:
#             st.metric("Scenario B ROI", f"{roi_b:.2f}")
