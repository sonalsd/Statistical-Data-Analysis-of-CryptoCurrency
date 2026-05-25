# Cryptocurrency Trend Prediction & Analysis

A machine learning project evaluating the predictability of cryptocurrency price trends using historical financial metrics. This repository implements and compares various regression models (for precise asset valuation) and classification architectures (to forecast market directionalities) on historical data.

## Project Overview
Predicting asset prices in highly volatile markets, such as cryptocurrencies, presents a significant challenge. This project leverages structured historical market data to build predictive workflows. 

Key analytical findings indicate that **classification models are significantly more effective than regression models** for predicting cryptocurrency price trends, because market movements exhibit structures influenced by trading volumes, capitalization sizes, and external market factors rather than clean continuous distributions.


## Dataset Insights
The workflow processes a multi-asset cryptocurrency dataset (`crypto_combine.csv`) consisting of **7,899 market records** across multiple prominent tokens (e.g., BTC, ETH, XRP) spanning chronological daily tracking.

### Features Available:
* `Crypto`: The asset identifier/ticker string (Categorical feature)
* `Date`: Chronological transaction timestamp
* `Open`: The opening price at the start of the daily trading window
* `High`: The peak market valuation reached during the daily window
* `Low`: The lowest valuation floor recorded during the day
* `Close`: The final execution price settlement at the end of the day


## Tech Stack & Dependencies
The notebook relies on standard data engineering, processing, and machine learning libraries:

* **Data Manipulation:** `pandas`, `numpy`
* **Statistical Analysis:** `scipy.stats`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning Suite:** `scikit-learn`
* **Imbalanced Class Handling:** `imbalanced-learn` (`SMOTE`, `RandomUnderSampler`)

To install the dependencies, you can execute:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy

## Workflow Architecture

### 1. Exploratory Data Analysis (EDA)
- Data ingestion and feature indexing  
- Dynamic profiling of dataset  
- Summary statistics extraction  

### 2. Data Preprocessing
- Feature scaling using `StandardScaler`  
- Encoding categorical features using `OneHotEncoder`  
- Handling class imbalance using:
  - SMOTE (Synthetic Minority Oversampling Technique)  
  - Random Undersampling  

### 3. Model Implementation

### Regression Pipeline
- Ridge Regression  
- Lasso Regression  
- Objective: Predict continuous closing prices  

### Classification Pipeline
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Objective: Predict market trend direction (Up / Down / Stable)

### 4. Evaluation & Validation
- Root Mean Squared Error (RMSE)  
- R² Score  
- Confusion Matrix  
- Classification Report  

## Conclusion

In our statistical analysis of cryptocurrency trends, we evaluated the performance of three machine learning models: **Logistic Regression**, **Decision Tree**, and **Random Forest**. The performance was assessed using **Accuracy** and **F1-score** to ensure a balanced evaluation of precision and recall.

Among the tested models, **Random Forest** emerged as the best-performing model with the highest **F1-score of 0.8878**. While Logistic Regression achieved the highest **accuracy (0.9035)**, Random Forest provided a better balance between precision and recall. The Decision Tree model showed the lowest performance among the three.

**Random Forest is the most suitable model** for analyzing cryptocurrency trends due to its strong balance between accuracy and F1-score. It can be effectively used for:
- Predictive analysis of crypto price movements  
- Risk assessment  
- Trading strategy development  

Future improvements may include hyperparameter tuning and the use of deep learning techniques to enhance prediction accuracy.

## How this Model Helps in Crypto Analysis

- Random Forest performed best due to its strong precision-recall balance.
- Cryptocurrency price movements are not fully random; they are influenced by:
  - Trading volume  
  - Market capitalization  
  - External economic and social factors  
- Classification models are more effective than regression models for predicting **trend direction (Up/Down/Stable)**.
- Classification-based predictions help traders and investors build **risk-aware trading strategies**.


## Opportunities

### Advanced Modeling
- Gradient Boosting (XGBoost, LightGBM, CatBoost)  
- Deep Learning models (LSTM, GRU) for time-series forecasting  

### Feature Engineering
- Technical indicators such as:
  - RSI (Relative Strength Index)  
  - MACD (Moving Average Convergence Divergence)  
- Sentiment analysis from news and social media  

### Deployment Potential
- Deploy model as:
  - Trading dashboard  
  - Real-time prediction system  
  - Decision support tool for investors  

### Dataset Expansion
- Extend dataset to include **2020–2025 crypto data**  
- Include multiple cryptocurrencies for better generalization  

### Automated Trading
- Integration with APIs such as:
  - Binance  
  - Coinbase  
- Enable algorithmic trading strategies  


## Future Research Directions

### Deep Learning Integration
- Use LSTM and Transformer models for sequence prediction  
- Incorporate real-time news sentiment and on-chain data  

### Model Interpretability
- Use SHAP (SHapley Additive exPlanations) to interpret model decisions  

### Blockchain Data Utilization
- Include:
  - Transaction volume  
  - Wallet activity  
  - Network growth metrics  

### Multimodal Data Fusion
- Combine:
  - Price trends  
  - Order book data  
  - Global economic indicators  
- Improve classification robustness and prediction quality  
