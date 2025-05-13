// scripts.js

const API_URL = 'http://127.0.0.1:5000';

// 1. Загрузка всех четырёх файлов
async function uploadAll() {
  const form = new FormData();
  ['births','deaths','migration','population'].forEach(id => {
    const inp = document.getElementById(id);
    if (inp.files.length) form.append(id, inp.files[0]);
  });
  const missing = ['births','deaths','migration','population']
    .filter(id => !form.has(id));
  if (missing.length) {
    alert('Не выбраны: ' + missing.join(', '));
    return;
  }
  try {
    const res = await fetch(`${API_URL}/api/upload`, {
      method: 'POST',
      body: form
    });
    if (!res.ok) throw new Error(await res.text());
    const js = await res.json();
    alert('Данные загружены:\n' + JSON.stringify(js.shapes, null, 2));
  } catch (e) {
    alert('Ошибка загрузки: ' + e.message);
  }
}
document.getElementById('uploadButton').addEventListener('click', uploadAll);

// 2. Обучение модели
async function trainModel() {
  const model = document.getElementById('modelSelect').value;
  if (!model) {
    alert('Сначала выберите модель');
    return;
  }

  try {
    const res = await fetch(`${API_URL}/api/model/train`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ model, params: {} })
    });
    if (!res.ok) throw new Error(await res.text());
    const js = await res.json();

    // Рендерим метрики
    function fmt(v,d=3,pct=false){
      if (v==null) return '-';
      const n = pct? v*100 : v;
      return n.toFixed(d)+(pct? '%':'');
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
    const trainImg = document.getElementById('populationPlot');
    if (trainImg) {
      trainImg.src = js.population_plot;
      trainImg.style.display = 'block';
    }

    // Разрешаем кнопку прогноза
    document.getElementById('forecastButton').disabled = false;

  } catch (e) {
    alert('Ошибка при обучении: ' + e.message);
  }
}
document.getElementById('trainButton').addEventListener('click', trainModel);

// 3. Ползунок для горизонта прогноза
const slider = document.getElementById('horizonSlider');
slider.addEventListener('input', () => {
  document.getElementById('horizonValue').textContent = slider.value;
});

// 4. Прогнозирование будущего
async function forecastFuture() {
  const horizon = parseInt(document.getElementById('horizonSlider').value, 10);

  try {
    const res = await fetch(`${API_URL}/api/model/forecast`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ model: 'sarimax', horizon })
    });
    if (!res.ok) throw new Error(await res.text());
    const js = await res.json();

    // Показываем финальный график с прогнозом
    const futureImg = document.getElementById('futurePlot');
    if (futureImg) {
      futureImg.src = js.image;
      futureImg.style.display = 'block';
    }

    // Строим сравнительную таблицу (с 2024)
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
}
document.getElementById('forecastButton').addEventListener('click', forecastFuture);
