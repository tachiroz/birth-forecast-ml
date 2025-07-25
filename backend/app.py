import os
import pickle
import warnings
import base64
import json
from datetime import datetime
from io import StringIO, BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np
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

BASE_DIR     = os.path.dirname(__file__)
DATA_FOLDER  = os.path.join(BASE_DIR, 'data')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')
RUNS_LOG     = os.path.join(BASE_DIR, 'runs.jsonl') 

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=''
)
CORS(app)

data_store = {}

# Rosstat forecasts
ROSSTAT_POP = {
    2024:1435064,2025:1426827,2026:1418382,2027:1409894,2028:1401407,
    2029:1393126,2030:1385054,2031:1377166,2032:1369463,2033:1361966,
    2034:1354742,2035:1347698,2036:1340824,2037:1334143,2038:1327678,
    2039:1321283,2040:1314929,2041:1308514,2042:1302170,2043:1295800,
    2044:1289493,2045:1283203,2046:1276946
}
ROSSTAT_BIRTHS = {
    2023:12282,2024:11497,2025:11310,2026:11203,2027:11158,
    2028:11293,2029:11447,2030:11563,2031:11688,2032:11878,
    2033:12114,2034:12250,2035:12423,2036:12631,2037:12860,
    2038:12975,2039:13045,2040:13070,2041:13084,2042:13041,
    2043:12987,2044:12972,2045:12934
}


def read_wrapped_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip().strip('"') for l in f if l.strip()]
    return pd.read_csv(StringIO("\n".join(lines)), sep=',')


def merge_full_df():
    for k in ('births','deaths','migration','population'):
        if k not in data_store:
            raise ValueError(f"Нет данных '{k}'. Сначала /api/upload")
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
    return df


def build_ml_model(name, params):
    if name == 'linear_regression':
        return LinearRegression()
    if name == 'random_forest':
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None)
        )
    if name == 'xgboost':
        return XGBRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.3),
            verbosity=0
        )
    if name == 'neural_network':
        return MLPRegressor(
            hidden_layer_sizes=tuple(params.get('hidden_layer_sizes', [100])),
            alpha=params.get('alpha', 0.0001),
            learning_rate_init=params.get('learning_rate_init', 0.001),
            max_iter=500
        )
    if name == 'svr':
        return SVR(
            kernel=params.get('kernel', 'rbf'),
            C=params.get('C', 1.0),
            gamma=params.get('gamma', 'scale')
        )
    raise ValueError(f"Неизвестная модель {name}")


def append_run_log(record: dict):
    """
    Дописывает одну JSON-строку в файл RUNS_LOG.
    Если файл отсутствует, создаёт его.
    """
    with open(RUNS_LOG, 'a', encoding='utf-8') as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_files():
    shapes = {}
    for key in ('births','deaths','migration','population'):
        f = request.files.get(key)
        if not f or not f.filename.lower().endswith('.csv'):
            return jsonify({'status':'error','message':f"Нет '{key}'"}), 400
        path = os.path.join(DATA_FOLDER, f"{key}.csv")
        f.save(path)
        data_store[key] = read_wrapped_csv(path)
        shapes[key] = data_store[key].shape
    return jsonify({'status':'success','shapes':shapes})


# === TRAIN population ===
@app.route('/api/model/train', methods=['POST'])
def train_population():
    payload    = request.get_json(force=True)
    model_name = payload['model']
    params     = payload.get('params', {})

    df = merge_full_df()
    tr = df[df['Year'] <= 2021]
    te = df[df['Year'] >= 2022]
    y_tr = tr['population']

    # 1) Обучение выбранной модели
    if model_name == 'prophet':
        ds = tr[['Year','population']].rename(columns={'Year':'ds','population':'y'})
        ds['ds'] = pd.to_datetime(ds['ds'], format='%Y')
        m = Prophet(
            changepoint_prior_scale=params.get('changepointPriorScale', 0.05),
            seasonality_mode=params.get('seasonalityMode', 'additive')
        )
        m.fit(ds)
        future = m.make_future_dataframe(periods=len(te), freq='Y')
        y_pred = m.predict(future)['yhat'].iloc[-len(te):].values

    elif model_name == 'sarimax':
        warnings.filterwarnings("ignore")
        order = tuple(params.get('order', [1,1,1]))
        seas  = tuple(params.get('seasonal_order', [0,0,0,2]))
        m = SARIMAX(
            y_tr,
            order=order,
            seasonal_order=seas,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        y_pred = m.predict(start=len(tr), end=len(tr) + len(te) - 1)

    else:
        Xc  = ['Year','births','deaths','migration']
        Xtr = tr[Xc]
        Xte = te[Xc]
        m   = build_ml_model(model_name, params)
        m.fit(Xtr, y_tr)
        y_pred = m.predict(Xte)

    # 2) Вычисление метрик
    mae  = mean_absolute_error(te['population'], y_pred)
    mse  = mean_squared_error(te['population'], y_pred)
    mape = mean_absolute_percentage_error(te['population'], y_pred)
    r2   = r2_score(te['population'], y_pred)
    mape = None if np.isnan(mape) else mape
    r2   = None if np.isnan(r2) else r2

    # 3) Записываем в лог (runs.jsonl)
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "target":    "population",
        "model":     model_name,
        "params":    params,
        "metrics":   {"mae": mae, "mse": mse, "mape": mape, "r2": r2}
    }
    append_run_log(record)

    # 4) Построение графика train + test
    plt.figure(figsize=(8,4))
    plt.plot(df['Year'], df['population'], 'k-', label='История')
    plt.plot(te['Year'], y_pred, '--o', color='tab:blue', label='Прогноз')
    plt.axvline(2021.5, linestyle='--', color='gray')
    plt.xlabel('Year'); plt.ylabel('Population'); plt.legend()
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

    # 5) Сохраняем модель на диск
    model_path = os.path.join(MODEL_FOLDER, f'model_population_{model_name}.pkl')
    with open(model_path, 'wb') as f_out:
        pickle.dump(m, f_out)

    # 6) Возвращаем ответ
    return jsonify({
        'status': 'success',
        'metrics': {'mae': mae, 'mse': mse, 'mape': mape, 'r2': r2},
        'plot': f'data:image/png;base64,{plot_b64}'
    })


# === FORECAST population ===
@app.route('/api/model/forecast', methods=['POST'])
def forecast_population():
    payload    = request.get_json(force=True)
    model_name = payload['model']
    params     = payload.get('params', {})
    horizon    = int(payload.get('horizon', 5))

    df = merge_full_df()
    start_year = 2024
    years = list(range(start_year, start_year + horizon))

    # 1) Прогнозирование
    if model_name == 'prophet':
        ds = df[['Year','population']].rename(columns={'Year':'ds','population':'y'})
        ds['ds'] = pd.to_datetime(ds['ds'], format='%Y')
        m = Prophet(
            changepoint_prior_scale=params.get('changepointPriorScale', 0.05),
            seasonality_mode=params.get('seasonalityMode', 'additive')
        )
        m.fit(ds)
        future = m.make_future_dataframe(periods=horizon, freq='Y')
        y_pred = m.predict(future)['yhat'].iloc[-horizon:]

    elif model_name == 'sarimax':
        model_path = os.path.join(MODEL_FOLDER, f'model_population_{model_name}.pkl')
        with open(model_path, 'rb') as f_in:
            m = pickle.load(f_in)
        y_pred = m.forecast(steps=horizon)

    else:
        model_path = os.path.join(MODEL_FOLDER, f'model_population_{model_name}.pkl')
        with open(model_path, 'rb') as f_in:
            m = pickle.load(f_in)
        last = df.iloc[-1]
        Xf = pd.DataFrame({
            'Year':      years,
            'births':    [last['births']]*horizon,
            'deaths':    [last['deaths']]*horizon,
            'migration': [last['migration']]*horizon
        })
        y_pred = m.predict(Xf)

    y_pred = list(y_pred)
    # 2) Выравниваем «старт» прогноза с последней точкой истории
    actual_last = df.loc[df['Year'] == start_year - 1, 'population'].iloc[0]
    shift = actual_last - y_pred[0]
    y_pred = [float(p + shift) for p in y_pred]

    # 3) Формируем таблицу сравнения
    table = []
    for yr, p in zip(years, y_pred):
        ros  = ROSSTAT_POP.get(yr)
        diff = None if ros is None else float(p - ros)
        pct  = None if (ros is None or ros == 0) else diff / ros * 100
        table.append({
            'year':    yr,
            'model':   p,
            'rosstat': ros,
            'diff':    diff,
            'diff_pct': pct
        })

    # 4) Построение графика (history + model + rosstat)
    plt.figure(figsize=(10,5))
    plt.plot(df['Year'], df['population'], 'k-', label='История')
    plt.plot(years, y_pred, '--o', color='tab:blue', label='Прогноз')
    plt.plot(years,
             [ROSSTAT_POP.get(y, np.nan) for y in years],
             '--o', color='tab:red',  label='Rosstat')
    plt.axvline(2021.5, linestyle='--', color='gray')
    plt.xlabel('Year'); plt.ylabel('Population'); plt.legend(); plt.grid(True)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')

    return jsonify({'image': f'data:image/png;base64,{img_b64}', 'table': table})


# === TRAIN births ===
@app.route('/api/model/train_births', methods=['POST'])
def train_births():
    payload    = request.get_json(force=True)
    model_name = payload['model']
    params     = payload.get('params', {})

    df = merge_full_df()
    tr = df[df['Year'] <= 2021]
    te = df[df['Year'] >= 2022]
    y_tr = tr['births']

    # 1) Обучение выбранной модели
    if model_name == 'prophet':
        ds = tr[['Year','births']].rename(columns={'Year':'ds','births':'y'})
        ds['ds'] = pd.to_datetime(ds['ds'], format='%Y')
        m = Prophet(
            changepoint_prior_scale=params.get('changepointPriorScale', 0.05),
            seasonality_mode=params.get('seasonalityMode', 'additive')
        )
        m.fit(ds)
        future = m.make_future_dataframe(periods=len(te), freq='Y')
        y_pred = m.predict(future)['yhat'].iloc[-len(te):].values

    elif model_name == 'sarimax':
        warnings.filterwarnings("ignore")
        order = tuple(params.get('order', [1,1,1]))
        seas  = tuple(params.get('seasonal_order', [0,0,0,2]))
        m = SARIMAX(
            y_tr,
            order=order,
            seasonal_order=seas,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        y_pred = m.predict(start=len(tr), end=len(tr) + len(te) - 1)

    else:
        Xc  = ['Year','deaths','migration','population']
        Xtr = tr[Xc]
        Xte = te[Xc]
        m   = build_ml_model(model_name, params)
        m.fit(Xtr, y_tr)
        y_pred = m.predict(Xte)

    # 2) Вычисление метрик
    mae  = mean_absolute_error(te['births'], y_pred)
    mse  = mean_squared_error(te['births'], y_pred)
    mape = mean_absolute_percentage_error(te['births'], y_pred)
    r2   = r2_score(te['births'], y_pred)
    mape = None if np.isnan(mape) else mape
    r2   = None if np.isnan(r2) else r2

    # 3) Запись в лог (runs.jsonl)
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "target":    "births",
        "model":     model_name,
        "params":    params,
        "metrics":   {"mae": mae, "mse": mse, "mape": mape, "r2": r2}
    }
    append_run_log(record)

    # 4) Построение графика train + test
    plt.figure(figsize=(8,4))
    plt.plot(df['Year'], df['births'], 'k-', label='История')
    plt.plot(te['Year'], y_pred, '--o', color='tab:blue', label='Прогноз')
    plt.axvline(2021.5, linestyle='--', color='gray')
    plt.xlabel('Year'); plt.ylabel('Births'); plt.legend()
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

    # 5) Сохраняем модель на диск
    model_path = os.path.join(MODEL_FOLDER, f'model_births_{model_name}.pkl')
    with open(model_path, 'wb') as f_out:
        pickle.dump(m, f_out)

    # 6) Возвращаем ответ
    return jsonify({
        'status': 'success',
        'metrics': {'mae': mae, 'mse': mse, 'mape': mape, 'r2': r2},
        'plot': f'data:image/png;base64,{plot_b64}'
    })


# === FORECAST births ===
@app.route('/api/model/forecast_births', methods=['POST'])
def forecast_births():
    payload    = request.get_json(force=True)
    model_name = payload['model']
    params     = payload.get('params', {})
    horizon    = int(payload.get('horizon', 5))

    df = merge_full_df()
    start_year = 2023
    years      = list(range(start_year, start_year + horizon))

    # 1) Прогнозирование
    if model_name == 'prophet':
        ds = df[['Year','births']].rename(columns={'Year':'ds','births':'y'})
        ds['ds'] = pd.to_datetime(ds['ds'], format='%Y')
        m = Prophet(
            changepoint_prior_scale=params.get('changepointPriorScale', 0.05),
            seasonality_mode=params.get('seasonalityMode', 'additive')
        )
        m.fit(ds)
        future = m.make_future_dataframe(periods=horizon, freq='Y')
        y_pred = m.predict(future)['yhat'].iloc[-horizon:]

    elif model_name == 'sarimax':
        model_path = os.path.join(MODEL_FOLDER, f'model_births_sarimax.pkl')
        with open(model_path, 'rb') as f_in:
            m = pickle.load(f_in)
        y_pred = m.forecast(steps=horizon)

    else:
        model_path = os.path.join(MODEL_FOLDER, f'model_births_{model_name}.pkl')
        with open(model_path, 'rb') as f_in:
            m = pickle.load(f_in)
        last = df.iloc[-1]
        Xf = pd.DataFrame({
            'Year':      years,
            'deaths':    [last['deaths']]*horizon,
            'migration': [last['migration']]*horizon,
            'population': [last['population']]*horizon
        })
        y_pred = m.predict(Xf)

    y_pred = list(y_pred)
    # 2) Выравниваем «старт» прогноза с последней точкой истории
    actual_last = df.loc[df['Year'] == start_year - 1, 'births'].iloc[0]
    shift = actual_last - y_pred[0]
    y_pred = [float(p + shift) for p in y_pred]

    # 3) Формируем таблицу сравнения
    table = []
    for yr, p in zip(years, y_pred):
        ros  = ROSSTAT_BIRTHS.get(yr)
        diff = None if ros is None else float(p - ros)
        pct  = None if (ros is None or ros == 0) else diff / ros * 100
        table.append({
            'year':    yr,
            'model':   p,
            'rosstat': ros,
            'diff':    diff,
            'diff_pct': pct
        })

    # 4) Построение графика (history + model + rosstat)
    plt.figure(figsize=(10,5))
    plt.plot(df['Year'], df['births'], 'k-', label='История')
    plt.plot(years, y_pred, '--o', color='tab:blue', label='Прогноз')
    plt.plot(years,
             [ROSSTAT_BIRTHS.get(y, np.nan) for y in years],
             '--o', color='tab:red',  label='Rosstat')
    plt.axvline(2021.5, linestyle='--', color='gray')
    plt.xlabel('Year'); plt.ylabel('Births'); plt.legend(); plt.grid(True)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')

    return jsonify({'image': f'data:image/png;base64,{img_b64}', 'table': table})


if __name__ == '__main__':
    app.run(debug=True)
