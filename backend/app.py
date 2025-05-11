import os
import base64
from io import StringIO, BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

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

# === Папки ===
BASE_DIR     = os.path.dirname(__file__)
DATA_FOLDER  = os.path.join(BASE_DIR, 'data')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')
os.makedirs(DATA_FOLDER,  exist_ok=True)
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
    # проверяем загрузку
    for key in ('births','deaths','migration','population'):
        if key not in data_store:
            raise ValueError(f"Данные '{key}' не загружены")

    # агрегируем по годам
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

    # outer-слияние, чтобы взять весь период
    df = (df_b.merge(df_d, on='Year', how='outer')
             .merge(df_m, on='Year', how='outer')
             .merge(df_p, on='Year', how='outer')
             .sort_values('Year'))

    # заполняем пустые:
    df['births'].ffill(inplace=True)      # берем последнее известное
    df['deaths'].ffill(inplace=True)
    df['migration'].fillna(0, inplace=True)
    df['population'].ffill(inplace=True)
    df['population'].bfill(inplace=True)

    # рассчитываем birth_rate
    df['birth_rate'] = df['births'] / df['population'] * 1000

    # отрежем лишние годы, где родов не было в исходных данных
    max_birth_year = data_store['births']['Year'].max()
    df = df[df['Year'] <= max_birth_year]

    return df

def build_model(name, params):
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
    keys = ['births','deaths','migration','population']
    shapes = {}
    for k in keys:
        f = request.files.get(k)
        if not f or not f.filename.lower().endswith('.csv'):
            return jsonify({'status':'error','message':f'Нет "{k}" или не .csv'}),400
        path = os.path.join(DATA_FOLDER, f"{k}.csv")
        f.save(path)
        try:
            df = read_wrapped_csv(path)
        except Exception as e:
            return jsonify({'status':'error','message':str(e)}),400
        data_store[k] = df
        shapes[k] = df.shape
    return jsonify({'status':'success','shapes':shapes})

@app.route('/api/model/train', methods=['POST'])
def train_model():
    payload    = request.get_json(force=True)
    model_name = payload.get('model')
    params     = payload.get('params',{})

    df = merge_full_df()

    years = sorted(df['Year'].unique())
    split_i = int(len(years)*0.8)
    cutoff = years[split_i]

    train_df = df[df['Year'] <= cutoff]
    test_df  = df[df['Year'] >  cutoff]

    # — убираем births из фич, чтобы модель не «видела» целевую переменную
    X_train = train_df[['Year','deaths','migration','population']]
    y_train = train_df['birth_rate']
    X_test  = test_df[['Year','deaths','migration','population']]
    y_test  = test_df['birth_rate']

    model = build_model(model_name, params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # метрики
    import math
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    try:
        mape = mean_absolute_percentage_error(y_test, y_pred)
        if math.isnan(mape): mape = None
    except: mape = None
    try:
        r2  = r2_score(y_test, y_pred)
        if math.isnan(r2):  r2  = None
    except: r2 = None

    # — график birth_rate —
    plt.figure(figsize=(8,4))
    plt.plot(df['Year'], df['birth_rate'], color='black', label='История')
    plt.plot(test_df['Year'], y_pred,
             linestyle='--', marker='o', color='tab:blue', label='Прогноз')
    plt.axvline(cutoff+0.5, linestyle='--', color='gray')
    plt.xlabel('Year'); plt.ylabel('Birth rate'); plt.legend()
    buf1 = BytesIO(); plt.tight_layout(); plt.savefig(buf1, format='png'); plt.close()
    buf1.seek(0)
    plot_birth = base64.b64encode(buf1.read()).decode('utf-8')

    # — график population —
    plt.figure(figsize=(8,4))
    plt.plot(df['Year'], df['population'], color='black', label='История')
    plt.plot(test_df['Year'], test_df['population'],
             linestyle='--', marker='o', color='tab:green', label='Тест (факт)')
    plt.axvline(cutoff+0.5, linestyle='--', color='gray')
    plt.xlabel('Year'); plt.ylabel('Population'); plt.legend()
    buf2 = BytesIO(); plt.tight_layout(); plt.savefig(buf2, format='png'); plt.close()
    buf2.seek(0)
    plot_pop = base64.b64encode(buf2.read()).decode('utf-8')

    joblib.dump(model, os.path.join(MODEL_FOLDER, f'model_{model_name}.joblib'))

    return jsonify({
        'status':'success',
        'metrics':{'mae':mae,'mse':mse,'mape':mape,'r2':r2},
        'plot': f'data:image/png;base64,{plot_birth}',
        'population_plot': f'data:image/png;base64,{plot_pop}'
    })

if __name__ == '__main__':
    app.run(debug=True)
