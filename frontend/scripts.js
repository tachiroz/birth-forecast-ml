// scripts.js

const API_URL = 'http://127.0.0.1:5000';

// 1) Показ параметров в зависимости от модели
document.getElementById('modelSelect').addEventListener('change', () => {
  const model = document.getElementById('modelSelect').value;
  const pDiv  = document.getElementById('parameters');
  pDiv.innerHTML = '';

  if (model === 'decision_tree') {
    pDiv.innerHTML = `
      <label for="maxDepth">Максимальная глубина дерева</label>
      <input type="number" id="maxDepth" placeholder="Введите максимальную глубину">
    `;
  } else if (model === 'random_forest') {
    pDiv.innerHTML = `
      <label for="numTrees">Количество деревьев</label>
      <input type="number" id="numTrees" placeholder="Введите количество деревьев">
      <label for="maxDepthRF">Макс. глубина</label>
      <input type="number" id="maxDepthRF" placeholder="Введите максимальную глубину">
    `;
  } else if (model === 'neural_network') {
    pDiv.innerHTML = `
      <label for="layers">Слои</label>
      <input type="number" id="layers" placeholder="Кол-во слоев">
      <label for="neurons">Нейроны в слое</label>
      <input type="number" id="neurons" placeholder="Нейронов">
      <label for="learningRate">LR</label>
      <input type="number" id="learningRate" step="0.0001" placeholder="Скорость обучения">
    `;
  } else if (model === 'linear_regression') {
    pDiv.innerHTML = `<label>Линейная регрессия не требует параметров</label>`;
  }
});

// 2) Загрузка CSV
async function uploadAll() {
  const form = new FormData();
  ['births','deaths','migration','population'].forEach(key => {
    const inp = document.getElementById(key);
    if (inp.files.length) form.append(key, inp.files[0]);
  });
  const missing = ['births','deaths','migration','population'].filter(k => !form.has(k));
  if (missing.length) {
    return alert('Не выбраны файлы: ' + missing.join(', '));
  }
  try {
    const res = await fetch(`${API_URL}/api/upload`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const js = await res.json();
    if (js.status !== 'success') throw new Error(js.message);
    alert('Данные загружены:\n' + JSON.stringify(js.shapes, null, 2));
  } catch (err) {
    console.error(err);
    alert('Ошибка загрузки: ' + err.message);
  }
}
document.getElementById('uploadButton').addEventListener('click', uploadAll);

// 3) Обучение модели и отображение результатов
async function trainModel() {
  const model = document.getElementById('modelSelect').value;
  if (!model) return alert('Сначала выберите модель');

  const params = {};
  if (model === 'decision_tree') {
    params.maxDepth = parseInt(document.getElementById('maxDepth').value, 10);
  } else if (model === 'random_forest') {
    params.numTrees = parseInt(document.getElementById('numTrees').value, 10);
    params.maxDepth = parseInt(document.getElementById('maxDepthRF').value, 10);
  } else if (model === 'neural_network') {
    params.layers       = parseInt(document.getElementById('layers').value, 10);
    params.neurons      = parseInt(document.getElementById('neurons').value, 10);
    params.learningRate = parseFloat(document.getElementById('learningRate').value);
  }

  try {
    const res = await fetch(`${API_URL}/api/model/train`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ model, params })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => null);
      throw new Error(err?.message || res.statusText);
    }
    const js = await res.json();

    // форматирование метрик, безопасно для null
    function fmt(val, digits=3, isPct=false) {
      if (val === null || val === undefined) return '-';
      const num = isPct ? val * 100 : val;
      return num.toFixed(digits) + (isPct ? '%' : '');
    }

    document.getElementById('metricsContainer').innerHTML = `
      <table class="metrics-table">
        <thead>
          <tr><th>Метрика</th><th>Значение</th></tr>
        </thead>
        <tbody>
          <tr><td>MAE</td><td>${fmt(js.metrics.mae)}</td></tr>
          <tr><td>MSE</td><td>${fmt(js.metrics.mse)}</td></tr>
          <tr><td>MAPE</td><td>${fmt(js.metrics.mape,2,true)}</td></tr>
          <tr><td>R²</td><td>${fmt(js.metrics.r2)}</td></tr>
        </tbody>
      </table>
    `;

    // показываем графики
    const fImg = document.getElementById('forecastPlot');
    const pImg = document.getElementById('populationPlot');
    fImg.src = js.plot;
    pImg.src = js.population_plot;
    fImg.style.display = 'block';
    pImg.style.display = 'block';

  } catch (err) {
    console.error(err);
    alert('Ошибка при обучении: ' + err.message);
  }
}
document.getElementById('trainButton').addEventListener('click', trainModel);
