import os
import json
import pandas as pd

# Путь к JSONL-логу
LOG_FILE = os.path.join(os.path.dirname(__file__), 'runs.jsonl')

# CSV, в который сохраняем лучшие метрики
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), 'best_runs.csv')

def load_runs(log_path):
    """
    Шаблон JSON строк
      {
        "timestamp": "...",
        "target": "population" или "births",
        "model":    "<название модели>",
        "params":   { ... },
        "metrics":  { "mae": ..., "mse": ..., "mape": ..., "r2": ... }
      }
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"{log_path}")
    records = []
    with open(log_path, encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                print(f"Ошибка! Не получилось распарсить строку:\n{line[:200]}…")
    return pd.DataFrame(records)


def find_best_per_model(df):
    """
    Из DataFrame со столбцами ['timestamp','target','model','params','metrics']
    выбор для каждой пары (target, model) записи с минимальным metrics['mape']
    На выходе возвращение нового DataFrame с колонками:
      [ 'target', 'model', 'params', 'mae', 'mse', 'mape', 'r2' ]
    """
    # Разбиение словаря metrics во вложенные колонки
    metrics_df = pd.json_normalize(df['metrics']).add_prefix('m_')
    df_expanded = pd.concat([df[['target', 'model', 'params']], metrics_df], axis=1)

    # Группирование комбинаций по (target, model)
    best_rows = []
    for (target, model), group in df_expanded.groupby(['target', 'model']):
        # Выбор той строки, у которой самое маленькое значение m_mape
        group_valid = group.dropna(subset=['m_mape'])
        if group_valid.empty:
            continue
        idx_min = group_valid['m_mape'].idxmin()
        best_rows.append(group.loc[idx_min])

    if not best_rows:
        return pd.DataFrame(columns=['target','model','params','m_mae','m_mse','m_mape','m_r2'])

    best_df = pd.DataFrame(best_rows)
    best_df = best_df[['target','model','params','m_mae','m_mse','m_mape','m_r2']]
    best_df = best_df.rename(columns={
        'm_mae': 'MAE',
        'm_mse': 'MSE',
        'm_mape': 'MAPE',
        'm_r2': 'R2'
    })
    return best_df.reset_index(drop=True)


def main():
    # Загрузка всех логов
    df_runs = load_runs(LOG_FILE)
    if df_runs.empty:
        print("Нет ни одной записи в runs.jsonl")
        return

    # Поиск лучших логов по MAPE для каждой комбинации
    best_df = find_best_per_model(df_runs)
    if best_df.empty:
        return

    # Вывод в консоль
    pd.set_option('display.max_colwidth', None)
    print("\nЛучшие запуски (минимальный MAPE) по каждой комбинации (target, model)")
    print(best_df.to_string(index=False))
    print("\nСохранение датасета в файл:\n", OUTPUT_CSV)

    # Сохранение в CSV
    best_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
