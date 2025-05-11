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

# Для SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX

# === Конфигурация путей ===
BASE_DIR     = os.path.dirname(__file__)
DATA_FOLDER  = os.path.join(BASE_DIR, 'data')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')

os.makedirs(DATA_FOLDER,  exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# === Flask ===
app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=''
)
CORS(app)

# в памяти храним оригинальные DataFrame
data_store = {}

# === Утилиты для чтения данных ===
def read_wrapped_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip().strip('"') for l in f if l.strip()]
    return pd.read_csv(StringIO("\n".join(lines)), sep=',')

def merge_full_df():
    # проверяем, что всё загружено
    for k in ('births','deaths','migration','population'):
        if k not in data_store:
            raise ValueError(f"Нет данных '{k}'. Сначала вызовите /api/upload")

    # агрегируем по годам
    births = (data_store['births']
              .rename(columns={'Birth':'births','birth':'births'})
              .groupby('Year', as_index=False)['births'].sum())
    deaths = (data_store['deaths']
              .rename(columns={'Died':'deaths','died':'deaths'})
              .groupby('Year', as_index=False)['deaths'].sum())
    mig    = (data_store['migration']
              .groupby('Year', as_index=False)
              .agg(come=('M_come','sum'), out=('M_out','sum')))
    mig['migration'] = mig['come'] - mig['out']
    mig = mig[['Year','migration']]
    pop    = (data_store['population']
              .rename(columns={'Population':'population'})
              .groupby('Year', as_index=False)['population'].sum())

    # outer-слияние, чтобы взять весь период (с 1990 до самого конца)
    df = (births
          .merge(deaths, on='Year', how='outer')
          .merge(mig,    on='Year', how='outer')
          .merge(pop,    on='Year', how='outer')
          .sort_values('Year'))

    # заполняем пропуски
    df['births'].ffill(inplace=True)
    df['deaths'].ffill(inplace=True)
    df['migration'].fillna(0, inplace=True)
    df['population'].ffill(inplace=True)
    df['population'].bfill(inplace=True)

    # рассчитываем birth_rate (пока не используется)
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
    raise ValueError('Неизвестная ML-модель')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_files():
    expected = ['births','deaths','migration','population']
    shapes = {}
    for k in expected:
        f = request.files.get(k)
        if not f or not f.filename.lower().endswith('.csv'):
            return jsonify({'status':'error','message':f'Нет "{k}" или не .csv'}), 400
        path = os.path.join(DATA_FOLDER, f"{k}.csv")
        f.save(path)
        df = read_wrapped_csv(path)
        data_store[k] = df
        shapes[k] = df.shape
    return jsonify({'status':'success','shapes':shapes})

@app.route('/api/model/train', methods=['POST'])
def train_model():
    payload    = request.get_json(force=True)
    model_name = payload.get('model')
    params     = payload.get('params', {})

    df = merge_full_df()

    # train/test split по годам: <=2021 / >=2022
    train_df = df[df['Year'] <= 2021].reset_index(drop=True)
    test_df  = df[df['Year'] >= 2022].reset_index(drop=True)

    # готовим exogenous для обоих типов моделей
    exog_cols = ['births','deaths','migration']
    # а целевая переменная — population
    y_train = train_df['population']
    y_test  = test_df['population']

    # --- SARIMAX ветка ---
    if model_name == 'sarimax':
        warnings.filterwarnings("ignore")
        best_aic = np.inf
        best_cfg = None

        # перебор p,q,P,Q
        for p in [0,1,2]:
            for q in [0,1,2]:
                for P in [0,1]:
                    for Q in [0,1]:
                        order = (p,1,q)
                        seas  = (P,1,Q,1)
                        try:
                            m = SARIMAX(
                                y_train,
                                exog=train_df[exog_cols],
                                order=order,
                                seasonal_order=seas,
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            ).fit(disp=False)
                            if m.aic < best_aic:
                                best_aic = m.aic
                                best_cfg = (order, seas)
                        except:
                            pass
        if best_cfg is None:
            best_cfg = ((1,1,1),(0,0,0,0))

        # финальная модель
        order, seas = best_cfg
        sarimax_model = SARIMAX(
            y_train,
            exog=train_df[exog_cols],
            order=order,
            seasonal_order=seas,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        y_pred = sarimax_model.predict(
            start=len(train_df),
            end=len(train_df)+len(test_df)-1,
            exog=test_df[exog_cols]
        )

        # метрики
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

    # --- ML-модели (вместо SARIMAX) прогноз населения ---
    else:
        X_train = train_df[['Year','deaths','migration','population']]
        X_test  = test_df[['Year','deaths','migration','population']]

        ml_model = build_ml_model(model_name, params)
        ml_model.fit(X_train, y_train)
        y_pred = ml_model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

    # конвертируем NaN→None
    import math
    if math.isnan(mape): mape = None
    if math.isnan(r2):   r2   = None

    # --- Строим только график population ---
    plt.figure(figsize=(8,4))
    plt.plot(df['Year'], df['population'], color='black', label='История')
    plt.plot(test_df['Year'], y_pred,
             linestyle='--', marker='o',
             color='tab:blue' if model_name=='sarimax' else 'tab:green',
             label=('SARIMAX прогноз' if model_name=='sarimax' else 'ML прогноз'))
    plt.axvline(2021.5, linestyle='--', color='gray')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.legend()
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    pop_plot = base64.b64encode(buf.read()).decode('utf-8')

    # сохраняем модель (для ML ветки)
    if model_name != 'sarimax':
        joblib.dump(ml_model, os.path.join(MODEL_FOLDER, f'model_{model_name}.joblib'))
    else:
        joblib.dump(sarimax_model, os.path.join(MODEL_FOLDER, f'model_sarimax.pkl'))

    return jsonify({
        'status':'success',
        'metrics': {'mae':mae,'mse':mse,'mape':mape,'r2':r2},
        'population_plot': f'data:image/png;base64,{pop_plot}'
    })

if __name__ == '__main__':
    app.run(debug=True)
