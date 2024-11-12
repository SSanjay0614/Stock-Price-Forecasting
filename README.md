# Stock-Price-Forecasting

## 1. Data Extraction and Preprocessing

I fetched the historical data of the EUR/INR exchange rate in 'Close', 'High', 'Low' terms using yfinance. As yfinance is a really apt package for the work, providing an easy to use API for getting data in most structured formats possible.After fetching the data, the initial inspection showed that 'Close', 'High', and 'Low' were object types. These were then converted to a float data type so that they can be used in numerical and forecasting operations. Missing values in 'Close', 'High', or 'Low' columns were filled forward.

## 2. Forecasting with ARIMA

Choice of Model (ARIMA): The ARIMA (Auto-Regressive Integrated Moving Average) model was selected for forecasting both one-day and one-week exchange rate values. ARIMA is a commonly used model for time series data due to its ability to capture trends and seasonality.
Application of ARIMA: The ARIMA model was fitted on the ‘Close’, ‘High’, and ‘Low’ columns for the specified period:
One-Day Forecast: We used a single-step forecast, predicting the immediate next day’s values.
One-Week Forecast: A multi-step forecast was generated to predict the closing prices over the next seven days. This allows us to identify trends or directional shifts in the exchange rate for the week ahead.

## 3. Computing Metrics
### Moving Average: 

Moving average for one-day as well as one-week values of forecast was computed. Moving average smoothes the fluctuations and provides a mean around which prices may deviate.

### Bollinger Bands: 

Calculated the value using a moving average centerline and an upper/lower band with an add on/off of two standard deviations.

### CCI: 

The CCI was calculated as (Typical Price−Moving Average)/Mean Absolute Deviation (MAD), providing a measure of the price deviation from its average.

## 4. Conclusion

After calculating all the above metrics, each indicator was used to produce BUY, SELL, or NEUTRAL signals.Moving Average: If the forecasted close was higher than the moving average, it was 'BUY'; if it was otherwise, it was 'SELL'. But if the forecasted close was very close to the moving average, then it was NEUTRAL.Bollinger Bands: If the forecasted close was above the upper band, it meant overvaluation, and thus a 'SELL' decision was taken. If it was below the lower band, then undervaluation was indicated, and thus a 'BUY' decision was taken. Otherwise, it is NEUTRAL.CCI: Signals that indicated CCI is OVERBOUGHT were taken at the point when values of CCI rise beyond +100. Signals 'BUY' when the value of CCI goes beyond -100. In between +100 to -100 of CCI values NEUTRAL was considered.


