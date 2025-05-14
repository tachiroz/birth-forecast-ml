// scripts.js

const API_URL = 'http://127.0.0.1:5000';

// === Табы ===
document.getElementById('tab-pop').onclick = () => {
  document.getElementById('tab-pop').classList.add('active');
  document.getElementById('tab-births').classList.remove('active');
  document.getElementById('panel-pop').classList.add('active');
  document.getElementById('panel-births').classList.remove('active');
};
document.getElementById('tab-births').onclick = () => {
  document.getElementById('tab-births').classList.add('active');
  document.getElementById('tab-pop').classList.remove('active');
  document.getElementById('panel-births').classList.add('active');
  document.getElementById('panel-pop').classList.remove('active');
};

// === Загрузка CSV (общая) ===
document.getElementById('uploadButton').onclick = async () => {
  const form = new FormData();
  ['births','deaths','migration','population'].forEach(id => {
    const inp = document.getElementById(id);
    if (inp.files.length) form.append(id, inp.files[0]);
  });
  const missing = ['births','deaths','migration','population'].filter(id => !form.has(id));
  if (missing.length) {
    return alert('Не выбраны: ' + missing.join(', '));
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

// === Параметры моделей ===
function initParams(tab) {
  const sel = document.getElementById('modelSelect'+tab);
  const dst = document.getElementById('parameters'+tab);
  sel.onchange = () => {
    const m = sel.value;
    dst.innerHTML = '';
    switch (m) {
      case 'linear_regression':
        dst.innerHTML = `<em>Нет доп. параметров.</em>`;
        break;

      case 'prophet':
        dst.innerHTML = `
          <label>Changepoint Prior Scale</label>
          <input type="number" id="changepointPriorScale${tab}" step="0.01" value="0.05">
          <label>Seasonality Mode</label>
          <select id="seasonalityMode${tab}">
            <option>additive</option>
            <option>multiplicative</option>
          </select>
        `;
        break;

      case 'random_forest':
        dst.innerHTML = `
          <label>n_estimators</label>
          <input type="number" id="rfEstimators${tab}" value="100">
          <label>max_depth</label>
          <input type="number" id="rfMaxDepth${tab}" placeholder="None">
        `;
        break;

      case 'xgboost':
        dst.innerHTML = `
          <label>n_estimators</label>
          <input type="number" id="xgbEstimators${tab}" value="100">
          <label>max_depth</label>
          <input type="number" id="xgbMaxDepth${tab}" value="6">
          <label>learning_rate</label>
          <input type="number" step="0.01" id="xgbLR${tab}" value="0.3">
        `;
        break;

      case 'neural_network':
        dst.innerHTML = `
          <label>hidden_layer_sizes (напр. "100,50")</label>
          <input type="text" id="nnHidden${tab}" value="100">
          <label>alpha</label>
          <input type="number" step="0.0001" id="nnAlpha${tab}" value="0.0001">
          <label>learning_rate_init</label>
          <input type="number" step="0.0001" id="nnLR${tab}" value="0.001">
        `;
        break;

      case 'svr':
        dst.innerHTML = `
          <label>kernel</label>
          <select id="svrKernel${tab}">
            <option>rbf</option><option>linear</option><option>poly</option>
          </select>
          <label>C</label>
          <input type="number" step="0.1" id="svrC${tab}" value="1.0">
          <label>gamma</label>
          <input type="text" id="svrGamma${tab}" value="scale">
        `;
        break;

      case 'sarimax':
        dst.innerHTML = `
          <label>order (p,d,q)</label><br>
          <input type="number" id="sarimaxP${tab}" value="1" min="0">,
          <input type="number" id="sarimaxD${tab}" value="1" min="0">,
          <input type="number" id="sarimaxQ${tab}" value="1" min="0"><br>
          <label>seasonal_order (P,D,Q,s)</label><br>
          <input type="number" id="sarimaxSP${tab}" value="0" min="0">,
          <input type="number" id="sarimaxSD${tab}" value="0" min="0">,
          <input type="number" id="sarimaxSQ${tab}" value="0" min="0">,
          <input type="number" id="sarimaxS${tab}"  value="2" min="2">
        `;
        break;
    }
  };
}

// === Обучение модели ===
function initTrain(tab, endpoint, plotId, metricsId, trainBtnId, forecastBtnId) {
  initParams(tab);
  document.getElementById(trainBtnId).onclick = async () => {
    const model = document.getElementById('modelSelect'+tab).value;
    if (!model) {
      alert('Сначала выберите модель');
      return;
    }

    // --- сбор параметров ---
    const params = {};
    if (model === 'prophet') {
      params.changepointPriorScale = parseFloat(
        document.getElementById(`changepointPriorScale${tab}`).value
      );
      params.seasonalityMode = document.getElementById(`seasonalityMode${tab}`).value;
    }
    else if (model === 'random_forest') {
      params.n_estimators = parseInt(
        document.getElementById(`rfEstimators${tab}`).value, 10
      );
      const md = document.getElementById(`rfMaxDepth${tab}`).value;
      params.max_depth = md ? parseInt(md,10) : null;
    }
    else if (model === 'xgboost') {
      params.n_estimators  = parseInt(
        document.getElementById(`xgbEstimators${tab}`).value, 10
      );
      params.max_depth     = parseInt(
        document.getElementById(`xgbMaxDepth${tab}`).value, 10
      );
      params.learning_rate = parseFloat(
        document.getElementById(`xgbLR${tab}`).value
      );
    }
    else if (model === 'neural_network') {
      params.hidden_layer_sizes = 
        document.getElementById(`nnHidden${tab}`).value
          .split(',').map(s=>parseInt(s.trim(),10));
      params.alpha              = parseFloat(
        document.getElementById(`nnAlpha${tab}`).value
      );
      params.learning_rate_init = parseFloat(
        document.getElementById(`nnLR${tab}`).value
      );
    }
    else if (model === 'svr') {
      params.kernel = document.getElementById(`svrKernel${tab}`).value;
      params.C      = parseFloat(
        document.getElementById(`svrC${tab}`).value
      );
      params.gamma  = document.getElementById(`svrGamma${tab}`).value;
    }
    else if (model === 'sarimax') {
      let s = parseInt(document.getElementById(`sarimaxS${tab}`).value,10);
      if (s < 2) s = 2;
      params.order = [
        parseInt(document.getElementById(`sarimaxP${tab}`).value,10),
        parseInt(document.getElementById(`sarimaxD${tab}`).value,10),
        parseInt(document.getElementById(`sarimaxQ${tab}`).value,10)
      ];
      params.seasonal_order = [
        parseInt(document.getElementById(`sarimaxSP${tab}`).value,10),
        parseInt(document.getElementById(`sarimaxSD${tab}`).value,10),
        parseInt(document.getElementById(`sarimaxSQ${tab}`).value,10),
        s
      ];
    }
    // (linear_regression — без параметров)

    // --- запрос на тренировку ---
    try {
      const res = await fetch(`${API_URL}/${endpoint}`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ model, params })
      });
      if (!res.ok) throw new Error(await res.text());
      const js = await res.json();

      // --- вывод метрик ---
      function fmt(v,d=3,pct=false){
        if (v==null) return '-';
        const n = pct ? v*100 : v;
        return n.toFixed(d) + (pct? '%' : '');
      }
      document.getElementById(metricsId).innerHTML = `
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

      // --- показать график train+test ---
      const img = document.getElementById(plotId);
      img.src = js.plot;
      img.style.display = 'block';

      document.getElementById(forecastBtnId).disabled = false;
    } catch (e) {
      alert('Ошибка при обучении: ' + e.message);
    }
  };
}

// === Прогнозирование ===
function initForecast(
  tab, endpoint,
  futurePlotId, tableId,
  sliderId, sliderValId,
  forecastBtnId
) {
  const slider = document.getElementById(sliderId);
  slider.oninput = () => {
    document.getElementById(sliderValId).textContent = slider.value;
  };
  document.getElementById(forecastBtnId).onclick = async () => {
    const model   = document.getElementById('modelSelect'+tab).value;
    const horizon = parseInt(document.getElementById(sliderId).value,10);
    try {
      const res = await fetch(`${API_URL}/${endpoint}`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ model, horizon })
      });
      if (!res.ok) throw new Error(await res.text());
      const js = await res.json();

      // график прогноза
      const img = document.getElementById(futurePlotId);
      img.src = js.image;
      img.style.display = 'block';

      // таблица сравнения
      let html = `<table class="metrics-table">
        <thead><tr><th>Year</th><th>Model</th><th>Rosstat</th><th>Diff</th><th>Diff %</th></tr></thead><tbody>`;
      js.table.forEach(r => {
        const dp = r.diff_pct==null? '-' : r.diff_pct.toFixed(0)+'%';
        html += `<tr>
          <td>${r.year}</td>
          <td>${r.model.toLocaleString()}</td>
          <td>${r.rosstat!=null?r.rosstat.toLocaleString():'-'}</td>
          <td>${r.diff!=null?r.diff.toLocaleString():'-'}</td>
          <td>${dp}</td>
        </tr>`;
      });
      html += '</tbody></table>';
      document.getElementById(tableId).innerHTML = html;

    } catch (e) {
      alert('Ошибка прогноза: ' + e.message);
    }
  };
}

// === Инициализация двух панелей ===
initTrain(
  'Pop',               // суффикс
  'api/model/train',   // endpoint
  'populationPlot',    // train-plot img id
  'metricsContainerPop',
  'trainButtonPop',
  'forecastButtonPop'
);
initForecast(
  'Pop',
  'api/model/forecast',
  'futurePlotPop',
  'compareTableContainerPop',
  'horizonSliderPop',
  'horizonValuePop',
  'forecastButtonPop'
);

initTrain(
  'Births',
  'api/model/train_births',
  'birthsPlot',
  'metricsContainerBirths',
  'trainButtonBirths',
  'forecastButtonBirths'
);
initForecast(
  'Births',
  'api/model/forecast_births',
  'futurePlotBirths',
  'compareTableContainerBirths',
  'horizonSliderBirths',
  'horizonValueBirths',
  'forecastButtonBirths'
);
