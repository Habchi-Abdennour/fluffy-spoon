import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import warnings


class TimeSeriesAnalysis:
    def __init__(self, dates, ordered):
        self.dates = pd.to_datetime(dates)
        self.ordered = np.array(ordered)
        self.df = pd.DataFrame({'date': self.dates, 'ordered': self.ordered})
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.set_index('date', inplace=True)

    def check_stationarity(self, column_name):
        result = adfuller(self.df[column_name], autolag='AIC')
        return 1 if result[1] < 0.05 else 0

    def preprocess_data(self):
        while self.check_stationarity('ordered') == 0:
            self.df['ordered'] = self.df['ordered'] - self.df['ordered'].shift(1)
            self.df['ordered'] = self.df['ordered'].fillna(self.df['ordered'].mean())

    def fit_arima_model(self, train_size=0.9):
        train_index = int(len(self.df) * train_size)
        train_data = self.df['ordered'][:train_index]

        model = auto_arima(y=train_data)
        order_par = model.order
        p, d, q = order_par
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()
        return model_fit

    def make_predictions(self, steps):
        model_fit = self.fit_arima_model()
        prediction = model_fit.forecast(steps=steps)
        return prediction










