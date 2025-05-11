// scripts.js

const API_URL = 'http://127.0.0.1:5000';

// 1) Динамика параметров
document.getElementById('modelSelect').addEventListener('change', () => {
  const m = document.getElementById('modelSelect').value;
  const p = document.getElementById('parameters');
  p.innerHTML = '';

  if (m === 'decision_tree') {
    p.innerHTML = `
      <label for="maxDepth">Максимальная глубина дерева</label>
      <input type="number" id="maxDepth" placeholder="…">
    `;
  } else if (m === 'random_forest') {
    p.innerHTML = `
      <label for="numTrees">Кол-во деревьев</label>
      <input type="number" id="numTrees" placeholder="…">
      <label for="maxDepthRF">Макс. глубина</label>
      <input type="number" id="maxDepthRF" placeholder="…">
    `;
  } else if (m === 'neural_network') {
    p.innerHTML = `
      <label for="layers">Слои</label>
      <input type="number" id="layers" placeholder="…">
      <label for="neurons">Нейроны</label>
      <input type="number" id="neurons" placeholder="…">
      <label for="learningRate">LR</label>
      <input type="number" id="learningRate" step="0.0001" placeholder="…">
    `;
  } else if (m === 'linear_regression') {
    p.innerHTML = `<label>Линейная регрессия без доп. параметров</label>`;
  } else if (m === 'sarimax') {
    p.innerHTML = `<label>SARIMAX подбирает параметры автоматически</label>`;
  }
});

// 2) Загрузка всех 4 файлов
async function uploadAll() {
  const form = new FormData();
  ['births','deaths','migration','population'].forEach(id => {
    const inp = document.getElementById(id);
    if (inp.files.length) form.append(id, inp.files[0]);
  });
  const missing = ['births','deaths','migration','population']
    .filter(id => !form.has(id));
  if (missing.length) {
    return alert('Не выбраны: ' + missing.join(', '));
  }
  try {
    const res = await fetch(`${API_URL}/api/upload`, {
      method: 'POST', body: form
    });
    if (!res.ok) throw new Error(await res.text());
    const j = await res.json();
    alert('Данные загружены:\n' + JSON.stringify(j.shapes, null,2));
  } catch(e) {
    alert('Ошибка загрузки: ' + e.message);
  }
}
document.getElementById('uploadButton')
  .addEventListener('click', uploadAll);

// 3) Обучение и отображение метрик + единственного графика
async function trainModel() {
  const model = document.getElementById('modelSelect').value;
  if (!model) return alert('Выберите модель');

  // собираем параметры
  const params = {};
  if (model === 'decision_tree') {
    params.maxDepth = +document.getElementById('maxDepth').value;
  } else if (model === 'random_forest') {
    params.numTrees = +document.getElementById('numTrees').value;
    params.maxDepth = +document.getElementById('maxDepthRF').value;
  } else if (model === 'neural_network') {
    params.layers       = +document.getElementById('layers').value;
    params.neurons      = +document.getElementById('neurons').value;
    params.learningRate = +document.getElementById('learningRate').value;
  }

  try {
    const res = await fetch(`${API_URL}/api/model/train`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model, params})
    });
    if (!res.ok) throw new Error(await res.text());
    const js = await res.json();

    // форматирование метрик
    function fmt(v,d=3,pct=false){
      if (v==null) return '-';
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

    // единственный график population
    const img = document.getElementById('populationPlot');
    img.src = js.population_plot;
    img.style.display = 'block';

  } catch(e) {
    alert('Ошибка при обучении: ' + e.message);
  }
}
document.getElementById('trainButton')
  .addEventListener('click', trainModel);
