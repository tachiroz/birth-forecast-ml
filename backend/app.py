# app.py

import os
import base64
from io import StringIO, BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import warnings

# matplotlib без GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
import joblib

from statsmodels.tsa.statespace.sarimax import SARIMAX

# === Папки ===
BASE_DIR     = os.path.dirname(__file__)
DATA_FOLDER  = os.path.join(BASE_DIR, 'data')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# === Flask ===
app = Flask(__name__,
            static_folder=FRONTEND_DIR,
            static_url_path='')
CORS(app)

data_store = {}

def read_wrapped_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip().strip('"') for l in f if l.strip()]
    return pd.read_csv(StringIO("\n".join(lines)), sep=',')

def merge_full_df():
    for key in ('births','deaths','migration','population'):
        if key not in data_store:
            raise ValueError(f"Нет данных '{key}'. Сначала /api/upload")
    df_b = (data_store['births']
            .rename(columns={'Birth':'births','birth':'births'})
            .groupby('Year', as_index=False)['births'].sum())
    df_d = (data_store['deaths']
            .rename(columns={'Died':'deaths','died':'deaths'})
            .groupby('Year', as_index=False)['deaths'].sum())
    df_m = (data_store['migration']
            .groupby('Year', as_index=False)
            .agg(come=('M_come','sum'), out=('M_out','sum')))
    df_m['migration'] = df_m['come'] - df_m['out']
    df_m = df_m[['Year','migration']]
    df_p = (data_store['population']
            .rename(columns={'Population':'population'})
            .groupby('Year', as_index=False)['population'].sum())

    df = (df_b.merge(df_d, on='Year', how='outer')
             .merge(df_m, on='Year', how='outer')
             .merge(df_p, on='Year', how='outer')
             .sort_values('Year'))
    df['births']     = df['births'].ffill()
    df['deaths']     = df['deaths'].ffill()
    df['migration']  = df['migration'].fillna(0)
    df['population'] = df['population'].ffill().bfill()
    df['birth_rate'] = df['births'] / df['population'] * 1000
    return df

def build_ml_model(name, params):
    if name == 'linear_regression':
        return LinearRegression(**params)
    if name == 'decision_tree':
        return DecisionTreeRegressor(max_depth=params.get('maxDepth'))
    if name == 'random_forest':
        return RandomForestRegressor(
            n_estimators=params.get('numTrees',100),
            max_depth=params.get('maxDepth')
        )
    if name == 'neural_network':
        layers  = params.get('layers',1)
        neurons = params.get('neurons',100)
        lr      = params.get('learningRate',0.001)
        return MLPRegressor(
            hidden_layer_sizes=(neurons,)*layers,
            learning_rate_init=lr,
            max_iter=500
        )
    raise ValueError('Неизвестная модель')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_files():
    expected = ['births','deaths','migration','population']
    shapes = {}
    for key in expected:
        f = request.files.get(key)
        if not f or not f.filename.lower().endswith('.csv'):
            return jsonify({'status':'error','message':f'Нет "{key}" или не .csv'}), 400
        path = os.path.join(DATA_FOLDER, f"{key}.csv")
        f.save(path)
        df = read_wrapped_csv(path)
        data_store[key] = df
        shapes[key] = df.shape
    return jsonify({'status':'success','shapes':shapes})

@app.route('/api/model/train', methods=['POST'])
def train_model():
    payload    = request.get_json(force=True)
    model_name = payload.get('model')
    params     = payload.get('params', {})

    df = merge_full_df()
    train_df = df[df['Year'] <= 2021]
    test_df  = df[df['Year'] >= 2022]

    y_train = train_df['population']
    y_test  = test_df['population']

    if model_name == 'sarimax':
        warnings.filterwarnings("ignore")
        model = SARIMAX(
            y_train,
            order=(1,1,1),
            seasonal_order=(0,0,0,0),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        y_pred = model.predict(start=len(train_df), end=len(train_df)+len(test_df)-1)
    else:
        X_train = train_df[['Year','deaths','migration','population']]
        X_test  = test_df[['Year','deaths','migration','population']]
        model = build_ml_model(model_name, params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = None if np.isnan(mape) else mape
    r2   = None if np.isnan(r2) else r2

    # график population
    plt.figure(figsize=(8,4))
    plt.plot(df['Year'], df['population'], color='black', label='История')
    plt.plot(test_df['Year'], y_pred,
             linestyle='--', marker='o', color='tab:blue', label='Прогноз')
    plt.axvline(2021.5, linestyle='--', color='gray')
    plt.xlabel('Year'); plt.ylabel('Population'); plt.legend()
    buf = BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png'); plt.close()
    buf.seek(0)
    pop_plot = base64.b64encode(buf.read()).decode('utf-8')

    joblib.dump(model, os.path.join(MODEL_FOLDER, f'model_{model_name}.pkl'))

    return jsonify({
        'status':'success',
        'metrics': {'mae':mae,'mse':mse,'mape':mape,'r2':r2},
        'population_plot': f'data:image/png;base64,{pop_plot}'
    })

ROSSTAT_FORECAST = {
    2024:1435064,2025:1426827,2026:1418382,2027:1409894,2028:1401407,
    2029:1393126,2030:1385054,2031:1377166,2032:1369463,2033:1361966,
    2034:1354742,2035:1347698,2036:1340824,2037:1334143,2038:1327678,
    2039:1321283,2040:1314929,2041:1308514,2042:1302170,2043:1295800,
    2044:1289493,2045:1283203,2046:1276946
}

@app.route('/api/model/forecast', methods=['POST'])
def forecast_future():
    data       = request.get_json(force=True)
    horizon    = int(data.get('horizon',5))

    df = merge_full_df()
    train = df[df['Year'] <= 2021]
    y_train = train['population']

    m = SARIMAX(
        y_train,
        order=(1,1,1),
        seasonal_order=(0,0,0,0),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    pred = m.forecast(steps=horizon)
    years = list(range(2022, 2022+horizon))

    ros_list = [ ROSSTAT_FORECAST.get(y, np.nan) for y in years ]

    table = []
    for yr, p, ros in zip(years, pred, ros_list):
        diff = None if np.isnan(ros) else float(p - ros)
        diff_pct = None if (np.isnan(ros) or ros == 0) else diff/ros*100
        table.append({'year':yr,'model':float(p),'rosstat':None if np.isnan(ros) else ros,'diff':diff,'diff_pct':diff_pct})

    plt.figure(figsize=(10,5))
    plt.plot(df['Year'], df['population'], color='black', label='История')
    plt.plot(years, pred, linestyle='--', marker='o', color='tab:blue', label='SARIMAX')
    plt.plot(years, ros_list, linestyle='--', marker='o', color='tab:red',  label='Rosstat')
    plt.axvline(2021.5, linestyle='--', color='gray')
    plt.xlabel('Year'); plt.ylabel('Population'); plt.legend(); plt.grid(True)
    buf = BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png'); plt.close()
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')

    return jsonify({'image':f'data:image/png;base64,{img}','table':table})

if __name__ == '__main__':
    app.run(debug=True)
