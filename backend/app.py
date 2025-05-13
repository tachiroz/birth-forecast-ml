# app.py

import os
import base64
import pickle
from io import StringIO, BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

# === Paths ===
BASE_DIR     = os.path.dirname(__file__)
DATA_FOLDER  = os.path.join(BASE_DIR, 'data')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')
os.makedirs(DATA_FOLDER,  exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app = Flask(__name__,
            static_folder=FRONTEND_DIR,
            static_url_path='')
CORS(app)

data_store = {}

ROSSTAT_FORECAST = {
    2024:1435064,2025:1426827,2026:1418382,2027:1409894,2028:1401407,
    2029:1393126,2030:1385054,2031:1377166,2032:1369463,2033:1361966,
    2034:1354742,2035:1347698,2036:1340824,2037:1334143,2038:1327678,
    2039:1321283,2040:1314929,2041:1308514,2042:1302170,2043:1295800,
    2044:1289493,2045:1283203,2046:1276946
}

def read_wrapped_csv(path):
    with open(path,'r',encoding='utf-8') as f:
        lines = [l.strip().strip('"') for l in f if l.strip()]
    return pd.read_csv(StringIO("\n".join(lines)), sep=',')

def merge_full_df():
    for key in ('births','deaths','migration','population'):
        if key not in data_store:
            raise ValueError(f"Нет данных '{key}'. Сначала /api/upload")
    df_b = (data_store['births']
            .rename(columns={'Birth':'births','birth':'births'})
            .groupby('Year',as_index=False)['births'].sum())
    df_d = (data_store['deaths']
            .rename(columns={'Died':'deaths','died':'deaths'})
            .groupby('Year',as_index=False)['deaths'].sum())
    df_m = (data_store['migration']
            .groupby('Year',as_index=False)
            .agg(come=('M_come','sum'), out=('M_out','sum')))
    df_m['migration'] = df_m['come'] - df_m['out']
    df_m = df_m[['Year','migration']]
    df_p = (data_store['population']
            .rename(columns={'Population':'population'})
            .groupby('Year',as_index=False)['population'].sum())

    df = (df_b.merge(df_d, on='Year', how='outer')
             .merge(df_m, on='Year', how='outer')
             .merge(df_p, on='Year', how='outer')
             .sort_values('Year'))
    df['births']     = df['births'].ffill()
    df['deaths']     = df['deaths'].ffill()
    df['migration']  = df['migration'].fillna(0)
    df['population'] = df['population'].ffill().bfill()
    return df

def build_ml_model(name, params):
    if name == 'linear_regression':
        return LinearRegression(**params)
    if name == 'random_forest':
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators',100),
            max_depth=params.get('max_depth',None)
        )
    if name == 'xgboost':
        return XGBRegressor(
            n_estimators=params.get('n_estimators',100),
            max_depth=params.get('max_depth',6),
            learning_rate=params.get('learning_rate',0.3),
            verbosity=0
        )
    if name == 'neural_network':
        return MLPRegressor(
            hidden_layer_sizes=tuple(params.get('hidden_layer_sizes',[100])),
            alpha=params.get('alpha',0.0001),
            learning_rate_init=params.get('learning_rate_init',0.001),
            max_iter=500
        )
    if name == 'svr':
        return SVR(
            kernel=params.get('kernel','rbf'),
            C=params.get('C',1.0),
            gamma=params.get('gamma','scale')
        )
    raise ValueError(f'Неизвестная ML-модель {name}')

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
        data_store[key] = read_wrapped_csv(path)
        shapes[key] = data_store[key].shape
    return jsonify({'status':'success','shapes':shapes})

@app.route('/api/model/train', methods=['POST'])
def train_model():
    payload    = request.get_json(force=True)
    model_name = payload.get('model')
    params     = payload.get('params', {})

    df       = merge_full_df()
    train_df = df[df['Year'] <= 2021]
    test_df  = df[df['Year'] >= 2022]
    y_train  = train_df['population']
    y_test   = test_df['population']

    # ===========================
    # 1) Обучаем выбранную модель
    # ===========================
    if model_name == 'prophet':
        ds = train_df[['Year','population']].rename(columns={'Year':'ds','population':'y'})
        ds['ds'] = pd.to_datetime(ds['ds'], format='%Y')
        m = Prophet(
            changepoint_prior_scale=params.get('changepointPriorScale',0.05),
            seasonality_mode=params.get('seasonalityMode','additive')
        )
        m.fit(ds)
        future   = m.make_future_dataframe(periods=len(test_df), freq='Y')
        forecast = m.predict(future)
        y_pred   = forecast['yhat'].iloc[-len(test_df):].values

    elif model_name == 'sarimax':
        warnings.filterwarnings("ignore")
        order = tuple(params.get('order', [1,1,1]))
        seas  = tuple(params.get('seasonal_order', [0,0,0,2]))
        m = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seas,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        y_pred = m.predict(
            start=len(train_df),
            end=len(train_df)+len(test_df)-1
        )

    else:
        X_cols  = ['Year','births','deaths','migration']
        X_train = train_df[X_cols]
        X_test  = test_df[X_cols]
        m = build_ml_model(model_name, params)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

    # ===========================
    # 2) Считаем метрики
    # ===========================
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = None if np.isnan(mape) else mape
    r2   = None if np.isnan(r2)   else r2

    # ===========================
    # 3) График train + test
    # ===========================
    plt.figure(figsize=(8,4))
    plt.plot(df['Year'], df['population'], color='black', label='История')
    plt.plot(test_df['Year'], y_pred,
             linestyle='--', marker='o', color='tab:blue', label='Прогноз')
    plt.axvline(2021.5, linestyle='--', color='gray')
    plt.xlabel('Year'); plt.ylabel('Population'); plt.legend()
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    pop_plot = base64.b64encode(buf.read()).decode('utf-8')

    # ===========================
    # 4) Сохраняем модель
    # ===========================
    model_path = os.path.join(MODEL_FOLDER, f'model_{model_name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(m, f)

    return jsonify({
        'status': 'success',
        'metrics': { 'mae': mae, 'mse': mse, 'mape': mape, 'r2': r2 },
        'population_plot': f'data:image/png;base64,{pop_plot}'
    })

@app.route('/api/model/forecast', methods=['POST'])
def forecast_future():
    data       = request.get_json(force=True)
    model_name = data.get('model')
    horizon    = int(data.get('horizon', 5))

    df      = merge_full_df()
    train   = df[df['Year'] <= 2021]
    y_train = train['population']

    # начинаем прогнозировать с 2024-го года
    start_year = 2024
    years      = list(range(start_year, start_year + horizon))

    # ===========================
    # 1) Загружаем модель и делаем прогноз
    # ===========================
    if model_name == 'prophet':
        with open(os.path.join(MODEL_FOLDER,'model_prophet.pkl'), 'rb') as f:
            m = pickle.load(f)
        last_year = train['Year'].max()
        # сколько шагов до 2024 и дальше
        steps = (start_year - last_year) + horizon
        future   = m.make_future_dataframe(periods=steps, freq='Y')
        forecast = m.predict(future)
        # берём последние 'horizon' значений
        y_pred = forecast['yhat'].iloc[-horizon:].values

    elif model_name == 'sarimax':
        with open(os.path.join(MODEL_FOLDER,'model_sarimax.pkl'), 'rb') as f:
            m = pickle.load(f)
        y_pred = m.forecast(steps=horizon)

    else:
        with open(os.path.join(MODEL_FOLDER,f'model_{model_name}.pkl'),'rb') as f:
            m = pickle.load(f)
        last = train.iloc[-1]
        Xf = pd.DataFrame({
            'Year':      years,
            'births':    [last['births']]   * horizon,
            'deaths':    [last['deaths']]   * horizon,
            'migration': [last['migration']]* horizon
        })
        y_pred = m.predict(Xf)

    # ===========================
    # 2) Собираем таблицу сравнения (все годы с 2024)
    # ===========================
    table = []
    for yr, p in zip(years, y_pred):
        ros      = ROSSTAT_FORECAST.get(yr)
        diff     = None if ros is None else float(p - ros)
        diff_pct = None if (ros is None or ros == 0) else diff/ros*100
        table.append({
            'year':    yr,
            'model':   float(p),
            'rosstat': ros,
            'diff':    diff,
            'diff_pct':diff_pct
        })

    # ===========================
    # 3) Финальный график
    # ===========================
    plt.figure(figsize=(10,5))
    plt.plot(df['Year'], df['population'], color='black', label='История')
    plt.plot(years, y_pred,
             linestyle='--', marker='o', color='tab:blue', label='Прогноз')
    plt.plot(years,
             [ROSSTAT_FORECAST.get(y, np.nan) for y in years],
             linestyle='--', marker='o', color='tab:red', label='Rosstat')
    plt.axvline(2021.5, linestyle='--', color='gray')
    plt.xlabel('Year'); plt.ylabel('Population'); plt.legend(); plt.grid(True)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')

    return jsonify({ 'image': f'data:image/png;base64,{img}', 'table': table })


if __name__ == '__main__':
    app.run(debug=True)
