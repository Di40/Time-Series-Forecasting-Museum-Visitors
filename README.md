# Time Series Forecasting of Museum Visitors
This repository showcases an application of concepts learned during the "Business, Economic, and Financial Data" course at the University of Padua to analyze time series data of the number of visitors in some Italian museums. 

## Objective

Predict monthly visitors for two Italian museums by leveraging time series forecasting techniques.

## Dataset

Manually crafted, using data from multiple sources:
- Main variable (museum visitors time series) from [Visit Piemonte](https://www.visitpiemonte-dmo.org/)
- External regressors: [Google Trends](https://trends.google.com/trends/), [Turin weather data](https://www.ilmeteo.it/), and some custom-built (# school holidays, COVID closures, and renovation).

## Baseline Models

- **Basic Baseline:** Predicts using the mean OR the same value as the last year, serving as a simple benchmark.
- **Advanced Baseline (SoTA):** Utilizes [TimeGPT](https://docs.nixtla.io/), representing the cutting-edge model for time series forecasting.

## Methods

- Holt-Winters' exponential smoothing with additive seasonality
- Generalized Additive Model (GAM)
- SARIMA
- SARIMAX
- Generalized Bass Model (GBM)
- Time-series linear regression
- Lasso regression
- Boosting (Gradient Boosting and XGBoost)

Also, combinations of methods are utilized, like GBM + SARIMAX for first modeling the trend, and then modeling the residuals.

## Hyperparameter Selection

Time-series cross validation in combination with AIC.

## Test Metrics

- RMSE
- MAPE
- AIC

## Results

### Museum of Cinema

Five models outperform both baselines, including: Exp. smoothing Holt Winters, TSLM, SARIMA, XGBoost, and Gradient Boosting.

![image](https://github.com/Di40/Time-Series-Forecasting-Museum-Visitors/assets/57565142/d8ea8b04-03ee-49b6-a220-318739ee05a6)

![image](https://github.com/Di40/Time-Series-Forecasting-Museum-Visitors/assets/57565142/572cf779-8921-4c98-a3a5-3c009b861bb3)


### Egyptian Museum

Eleven models outperform both baselines, with SARIMAX showing exceptional performance (0.271 RMSE vs 1.085 of TimeGPT).

![image](https://github.com/Di40/Time-Series-Forecasting-Museum-Visitors/assets/57565142/7ac4d913-cbac-4a8f-83e7-950a7859483c)

![image](https://github.com/Di40/Time-Series-Forecasting-Museum-Visitors/assets/57565142/5baa9388-0672-482e-8bab-60485d18464f)

## COVID lockdown - Effects on forecasting
Finally, performed analysis of the effects of COVID, trying to interpolate the outliers using two approaches:

1. Use a good forecasting model.
2. Replace each month using the historical monthly mean.

This didn't give much of an improvement of the previous models.

## Conclusion

![image](https://github.com/Di40/Time-Series-Forecasting-Museum-Visitors/assets/57565142/22dec01c-76f4-495b-9986-40f98960f585)


