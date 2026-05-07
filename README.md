# Demand Forecasting with XGBoost & Streamlit

This project implements an end-to-end machine learning solution for predicting product demand based on historical sales data, pricing, and promotional factors.

## 🚀 Project Overview
The goal of this project is to provide businesses with a tool to anticipate future demand, allowing for better inventory management and optimized pricing strategies. It includes a full data processing pipeline, a tuned XGBoost regressor, and a real-time prediction dashboard built with Streamlit.

## 🛠️ Tech Stack
- **Data Processing:** Python, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** XGBoost, Scikit-learn
- **Deployment:** Streamlit
- **Model Persistence:** Pickle

## 📊 Model Performance
The model was trained using `RandomizedSearchCV` to optimize hyperparameters. The current performance metrics are:
- **RMSE:** 35.60
- **MAE:** 27.13
- **R2 Score:** 0.43

*Key features used: Price, Discount, Inventory Level, Promotion, Competitor Pricing, and Category.*

## 💻 How to Run the App

1. **Clone the repository:**
   ```bash
   git clone https://github.com/URTD14/demand-forecasting_model_live.git
   cd demand-forecasting_model_live
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 📂 Repository Structure
- `app.py`: Streamlit application code.
- `demand_forecasting_model.pkl`: Trained XGBRegressor model.
- `label_encoders.pkl`: Encoders for categorical variables.
- `requirements.txt`: List of required Python packages.
- `demand_forecasting.csv`: (Optional) The dataset used for training.

## 🔮 Future Improvements
- Incorporate seasonality features (Month, Weekday) more deeply into the model.
- Test advanced time-series architectures like LSTMs or Prophet.
- Add more external features like local event data or economic indicators.
