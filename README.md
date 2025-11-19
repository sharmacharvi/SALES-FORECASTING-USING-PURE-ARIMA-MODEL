# SALES-FORECASTING-USING-PURE-ARIMA-MODEL

## Project Overview
This project focuses on forecasting daily sales using historical sales data. It involves exploratory data analysis (EDA), time series preprocessing, building an ARIMA model, evaluating its performance, and forecasting future sales. The primary goal is to predict future sales trends and identify areas for model improvement.

## Dataset
The analysis is based on the `train.csv` dataset, which contains sales information including `id`, `date`, `store_nbr`, `family`, `sales`, and `onpromotion` for various products across different stores.

## Exploratory Data Analysis (EDA) Highlights
*   **Data Characteristics**: The dataset initially had 3,000,888 entries. The `date` column was converted to datetime objects, and daily sales were aggregated into a `daily_sales` DataFrame, with missing dates imputed as zero.
*   **Sales Distribution**: Sales data showed a highly right-skewed distribution, indicating many zero sales events (e.g., holidays) and significant high-value outliers.
*   **Time Series Trends**: Total sales over time exhibited a clear upward trend and strong yearly seasonality, with notable dips around January 1st each year.
*   **Categorical Impact**: Significant variations in sales were observed across different `store_nbr` and `family` categories. 'BEVERAGES' and 'GROCERY I' were identified as top-performing product families.

## Time Series Analysis and Modeling
### Stationarity Testing
*   An Augmented Dickey-Fuller (ADF) test on the `daily_sales` series yielded a P-value of 0.0991, indicating that the series was **likely non-stationary**.

### ARIMA Model (Initial Attempt)
*   **Model Order**: Based on the ADF test and preliminary ACF/PACF plots, an initial ARIMA(1, 1, 1) model was chosen, employing first-order differencing (`d=1`) to address non-stationarity.
*   **Model Diagnostics**: The ARIMA(1, 1, 1) model, while trained, showed significant diagnostic issues:
    *   **Residual Autocorrelation**: Ljung-Box test indicated uncaptured autocorrelation (Prob(Q) = 0.00).
    *   **Non-Normal Residuals**: Jarque-Bera test showed non-normal residual distribution (Prob(JB) = 0.00).
    *   **Heteroskedasticity**: Heteroskedasticity test indicated non-constant variance (Prob(H) = 0.00).
    *   **Parameter Instability**: Warnings about a singular or near-singular covariance matrix suggested unstable parameter estimates.

## Model Evaluation and Forecasting Results
### Train/Test Split Evaluation
*   The `daily_sales` data was split into an 80% training set and 20% test set.
*   The ARIMA(1, 1, 1) model was re-trained on the training data.
*   **RMSE**: The Root Mean Squared Error (RMSE) on the test set was **158860.81**, highlighting a substantial deviation between predictions and actual sales. This indicates the model struggles to capture extreme fluctuations and zero-sales events.

### Future Sales Forecasting
*   The trained ARIMA model was used to forecast sales for the next 30 days, providing mean predictions along with 95% confidence intervals.

## Future Work and Improvements
To enhance forecasting accuracy and address the identified model limitations, the following steps are recommended:

1.  **SARIMA Model Exploration**: Given the strong seasonality, a Seasonal ARIMA (SARIMA) model should be explored to capture yearly or other periodic patterns more effectively.
2.  **Handling Outliers and Zero Sales**: Implement robust preprocessing techniques for handling zero sales (e.g., on New Year's Day) and high-value outliers. This might include imputation, robust statistical methods, or data transformations (e.g., log transformation).
3.  **Advanced Model Selection**: Investigate more sophisticated time series models beyond ARIMA/SARIMA, such as:
    *   **Prophet**: Especially useful for business time series with strong seasonal components and holiday effects.
    *   **XGBoost/LightGBM with Time Series Features**: Incorporate rich time-based features (lag features, rolling means, holiday indicators, store/family IDs) for machine learning models.
    *   **Neural Networks (LSTMs)**: Consider deep learning models for highly complex patterns and longer-term forecasting.
4.  **Incorporating External Features**: Integrate exogenous variables like `onpromotion` status, store characteristics (from `stores.csv`), and official holiday information to improve predictive power.
5.  **Residual Analysis and Model Refinement**: Continuously analyze model residuals (ACF/PACF plots, QQ plots) after each iteration to identify uncaptured patterns and guide further model adjustments.
6.  **Cross-Validation for Time Series**: Employ time series-specific cross-validation techniques (e.g., rolling origin validation) for more robust model evaluation and hyperparameter tuning.

## Setup and Usage
To run this project, you will need a Python environment with the following dependencies:

### Dependencies
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `statsmodels`
*   `scikit-learn`

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

Clone this repository:

```bash
git clone <repository-url>
cd <repository-name>
jupyter notebook
```

Then open the `your_notebook_name.ipynb` file.
