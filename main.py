
from fastapi import FastAPI
import requests,uvicorn
from class_arima import TimeSeriesAnalysis

app = FastAPI()


@app.get("/")
async def main():
    dates = [ 
        '2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01',
         '2019-05-01', '2019-06-01', '2019-07-01', '2019-08-01',
         '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01',
        '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01',
         '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01',
         '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01',
        '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01',
         '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01',
         '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01',
        '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01',
         '2022-05-01', '2022-06-01', '2022-07-01', '2022-08-01',
         '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01',
         '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01',
         '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01',
         '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01']

    ordered = [
        48, 96, 15, 65, 73, 13, 39, 95, 14, 88, 27, 17,
        35, 49, 27, 35, 73, 33, 49, 45, 65, 18, 17, 72,
        25, 29, 17, 35, 13, 43, 19, 65, 15, 28, 87, 37,
        45, 22, 87, 30, 73, 67, 0, 15, 5, 23, 10, 73,
        25, 65, 39, 29, 0, 56, 99, 77, 50, 50, 20, 75
        ]

    
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
