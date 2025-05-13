// scripts.js

const API_URL = 'http://127.0.0.1:5000';

// 1) Загрузка CSV
document.getElementById('uploadButton').onclick = async () => {
  const form = new FormData();
  ['births','deaths','migration','population'].forEach(id => {
    const inp = document.getElementById(id);
    if (inp.files.length) form.append(id, inp.files[0]);
  });
  const missing = ['births','deaths','migration','population'].filter(id => !form.has(id));
  if (missing.length) {
    alert('Не выбраны: ' + missing.join(', '));
    return;
  }
  try {
    const res = await fetch(`${API_URL}/api/upload`, { method:'POST', body: form });
    if (!res.ok) throw new Error(await res.text());
    const js = await res.json();
    alert('Данные загружены:\n' + JSON.stringify(js.shapes, null,2));
  } catch (e) {
    alert('Ошибка загрузки: ' + e.message);
  }
};

// 2) Показ гиперпараметров в зависимости от модели
document.getElementById('modelSelect').onchange = () => {
  const m = document.getElementById('modelSelect').value;
  const p = document.getElementById('parameters');
  p.innerHTML = '';
  switch (m) {
    case 'linear_regression':
      p.innerHTML = `<em>Нет дополнительных параметров.</em>`;
      break;

    case 'prophet':
      p.innerHTML = `
        <label>Changepoint Prior Scale</label>
        <input type="number" id="changepointPriorScale" step="0.01" value="0.05">
        <label>Seasonality Mode</label>
        <select id="seasonalityMode">
          <option>additive</option>
          <option>multiplicative</option>
        </select>
      `;
      break;

    case 'random_forest':
      p.innerHTML = `
        <label>n_estimators</label>
        <input type="number" id="rfEstimators" value="100">
        <label>max_depth</label>
        <input type="number" id="rfMaxDepth" value="">
      `;
      break;

    case 'xgboost':
      p.innerHTML = `
        <label>n_estimators</label>
        <input type="number" id="xgbEstimators" value="100">
        <label>max_depth</label>
        <input type="number" id="xgbMaxDepth" value="6">
        <label>learning_rate</label>
        <input type="number" step="0.01" id="xgbLR" value="0.3">
      `;
      break;

    case 'neural_network':
      p.innerHTML = `
        <label>hidden_layer_sizes (напр. "100,50")</label>
        <input type="text" id="nnHidden" value="100">
        <label>alpha</label>
        <input type="number" step="0.0001" id="nnAlpha" value="0.0001">
        <label>learning_rate_init</label>
        <input type="number" step="0.0001" id="nnLR" value="0.001">
      `;
      break;

    case 'svr':
      p.innerHTML = `
        <label>kernel</label>
        <select id="svrKernel">
          <option>rbf</option>
          <option>linear</option>
          <option>poly</option>
        </select>
        <label>C</label>
        <input type="number" step="0.1" id="svrC" value="1.0">
        <label>gamma</label>
        <input type="text" id="svrGamma" value="scale">
      `;
      break;

    case 'sarimax':
      p.innerHTML = `<em>Без гиперпараметров (будет использоваться (1,1,1)).</em>`;
      break;
  }
};

// 3) Обучение модели
document.getElementById('trainButton').onclick = async () => {
  const model = document.getElementById('modelSelect').value;
  if (!model) {
    alert('Сначала выберите модель');
    return;
  }

  // Собираем параметры
  const params = {};
  if (model === 'prophet') {
    params.changepointPriorScale = parseFloat(document.getElementById('changepointPriorScale').value);
    params.seasonalityMode       = document.getElementById('seasonalityMode').value;
  }
  if (model === 'random_forest') {
    params.n_estimators = parseInt(document.getElementById('rfEstimators').value, 10);
    const md = document.getElementById('rfMaxDepth').value;
    params.max_depth = md ? parseInt(md, 10) : null;
  }
  if (model === 'xgboost') {
    params.n_estimators  = parseInt(document.getElementById('xgbEstimators').value, 10);
    params.max_depth     = parseInt(document.getElementById('xgbMaxDepth').value, 10);
    params.learning_rate = parseFloat(document.getElementById('xgbLR').value);
  }
  if (model === 'neural_network') {
    params.hidden_layer_sizes = document.getElementById('nnHidden').value
      .split(',').map(s => parseInt(s.trim(),10));
    params.alpha              = parseFloat(document.getElementById('nnAlpha').value);
    params.learning_rate_init = parseFloat(document.getElementById('nnLR').value);
  }
  if (model === 'svr') {
    params.kernel = document.getElementById('svrKernel').value;
    params.C      = parseFloat(document.getElementById('svrC').value);
    params.gamma  = document.getElementById('svrGamma').value;
  }

  try {
    const res = await fetch(`${API_URL}/api/model/train`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ model, params })
    });
    if (!res.ok) throw new Error(await res.text());
    const js = await res.json();

    // Отображаем метрики
    function fmt(v,d=3,pct=false){
      if (v == null) return '-';
      const n = pct? v*100 : v;
      return n.toFixed(d) + (pct? '%' : '');
    }
    document.getElementById('metricsContainer').innerHTML = `
      <table class="metrics-table">
        <thead><tr><th>Метрика</th><th>Значение</th></tr></thead>
        <tbody>
          <tr><td>MAE</td><td>${fmt(js.metrics.mae)}</td></tr>
          <tr><td>MSE</td><td>${fmt(js.metrics.mse)}</td></tr>
          <tr><td>MAPE</td><td>${fmt(js.metrics.mape,2,true)}</td></tr>
          <tr><td>R²</td><td>${fmt(js.metrics.r2)}</td></tr>
        </tbody>
      </table>
    `;

    // Показываем график Train+Test
    const imgTrain = document.getElementById('populationPlot');
    imgTrain.src = js.population_plot;
    imgTrain.style.display = 'block';

    // Разрешаем кнопку прогноза
    document.getElementById('forecastButton').disabled = false;

  } catch (e) {
    alert('Ошибка при обучении: ' + e.message);
  }
};

// 4) Ползунок прогноза
const slider = document.getElementById('horizonSlider');
slider.oninput = () => {
  document.getElementById('horizonValue').textContent = slider.value;
};

// 5) Прогноз на будущее
document.getElementById('forecastButton').onclick = async () => {
  const model   = document.getElementById('modelSelect').value;
  const horizon = parseInt(document.getElementById('horizonSlider').value, 10);

  try {
    const res = await fetch(`${API_URL}/api/model/forecast`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ model, horizon })
    });
    if (!res.ok) throw new Error(await res.text());
    const js = await res.json();

    // Показываем график прогноза
    const imgFuture = document.getElementById('futurePlot');
    imgFuture.src = js.image;
    imgFuture.style.display = 'block';

    // Строим сравнительную таблицу
    let html = `<table class="metrics-table">
      <thead><tr>
        <th>Year</th><th>Model</th><th>Rosstat</th><th>Diff</th><th>Diff&nbsp;%</th>
      </tr></thead><tbody>`;
    js.table.forEach(r => {
      const dp = r.diff_pct == null ? '-' : r.diff_pct.toFixed(0) + '%';
      html += `<tr>
        <td>${r.year}</td>
        <td>${r.model.toLocaleString()}</td>
        <td>${r.rosstat != null ? r.rosstat.toLocaleString() : '-'}</td>
        <td>${r.diff != null ? r.diff.toLocaleString() : '-'}</td>
        <td>${dp}</td>
      </tr>`;
    });
    html += '</tbody></table>';
    document.getElementById('compareTableContainer').innerHTML = html;

  } catch (e) {
    alert('Ошибка прогноза: ' + e.message);
  }
};
