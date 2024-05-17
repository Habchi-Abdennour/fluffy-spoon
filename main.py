
from fastapi import FastAPI
import requests,uvicorn
from class_arima import TimeSeriesAnalysis

app = FastAPI()


@app.get("/")
async def main():
    dates = ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01',
         '2022-05-01', '2022-06-01', '2022-07-01', '2022-08-01',
         '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01',
         '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01',
         '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01',
         '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01']

    ordered = [45, 22, 87, 30, 75, 67, 90, 15, 50, 23, 010, 75,
           25, 50, 100, 29, 30, 50, 100, 75, 50, 150, 20, 75]
    return {'dates':dates,'ordered':ordered}




@app.get("/predict")
def read_root(url:str,value:int):
    read=requests.get(url)
    format=read.json()
    dates=format['dates']
    ordered=format['ordered']
    tsa = TimeSeriesAnalysis(dates, ordered)
    tsa.preprocess_data()
    predictions = tsa.make_predictions(value)

    return  {'dates': predictions.index.strftime('%Y-%m-%d').tolist(), 'ordered': predictions.tolist()}