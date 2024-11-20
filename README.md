# **Marketing Mix Modeling and Forecasting Tool**

This project implements a **Marketing Mix Modeling (MMM)** analysis to help businesses estimate the impact of their marketing activities and forecast sales or revenue based on historical data. The tool allows for **scenario planning** and **strategic decision-making** by integrating MMM predictions with external variables like seasonality and economic indicators.

---

## **Features**

1. **MMM Analysis**:
   - Builds a regression model to estimate the contribution of marketing channels (e.g., TV, Social Media, Digital) to sales.
   - Includes support for external factors like seasonality.

2. **Forecasting**:
   - Predicts future sales or revenue based on customizable marketing spends and external variables.

3. **Scenario Planning**:
   - Allows users to simulate different marketing mix scenarios to optimize strategy and budget allocation.
   - Visualizes the impact of different scenarios for informed decision-making.

---

## **Installation**

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required dependencies:
   ```
   python -m pip install -r requirements.txt
   ```

---

## Usage
1. Run MMM Analysis: Prepare your data or use the provided dummy dataset to build an MMM model:
    ```
    python mmm_analysis.py
    ```

2. Forecast Sales: Use the tool to predict future sales based on marketing spends and external variables:

    ```
    python forecast.py
    ```

3. Scenario Planning - Simulate multiple marketing mix scenarios and visualize their impact:

    ```
    streamlit run app.py 
    ```

---

## Inputs and Outputs

Input:

- Historical marketing spend data (e.g., TV, Social Media, Digital).
- Optional external variables (e.g., seasonality).

Output:

- Regression coefficients and performance metrics.
- Forecasted sales for future scenarios.
- Visualizations of scenarios and ROI.