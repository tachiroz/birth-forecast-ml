// scripts.js

const API_URL = 'http://127.0.0.1:5000';

document.getElementById('uploadButton').onclick = async () => {
  const form = new FormData();
  ['births','deaths','migration','population'].forEach(id => {
    const inp = document.getElementById(id);
    if (inp.files.length) form.append(id, inp.files[0]);
  });
  const missing = ['births','deaths','migration','population'].filter(id => !form.has(id));
  if (missing.length) return alert('Не выбраны: '+missing.join(', '));
  try {
    const res = await fetch(`${API_URL}/api/upload`, {method:'POST', body:form});
    if (!res.ok) throw new Error(await res.text());
    const j = await res.json();
    alert('Данные загружены:\n'+JSON.stringify(j.shapes,null,2));
  } catch(e) {
    alert('Ошибка загрузки: '+e);
  }
};

document.getElementById('trainButton').onclick = async () => {
  const model = document.getElementById('modelSelect').value;
  if (!model) return alert('Выберите модель');
  try {
    const res = await fetch(`${API_URL}/api/model/train`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model, params:{}})
    });
    if (!res.ok) throw new Error(await res.text());
    const js = await res.json();
    function fmt(v,d=2,pct=false){
      if (v==null) return '-';
      const n = pct? v*100 : v;
      return n.toFixed(d)+(pct?'%':'');
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
    const img = document.getElementById('futurePlot');
    img.src = js.population_plot;
    img.style.display = 'block';
    document.getElementById('forecastButton').disabled = false;
  } catch(e) {
    alert('Ошибка при обучении: '+e);
  }
};

const slider = document.getElementById('horizonSlider');
slider.oninput = () => document.getElementById('horizonValue').textContent = slider.value;

document.getElementById('forecastButton').onclick = async () => {
  const horizon = +document.getElementById('horizonSlider').value;
  try {
    const res = await fetch(`${API_URL}/api/model/forecast`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model:'sarimax', horizon})
    });
    if (!res.ok) throw new Error(await res.text());
    const js = await res.json();
    const img = document.getElementById('futurePlot');
    img.src = js.image;
    img.style.display = 'block';
    let html = `<table class="metrics-table">
      <thead><tr>
        <th>Year</th><th>Model</th><th>Rosstat</th><th>Diff</th><th>Diff&nbsp;%</th>
      </tr></thead><tbody>`;
    js.table.forEach(r => {
      const dp = r.diff_pct==null?'-': r.diff_pct.toFixed(0)+'%';
      html += `<tr>
        <td>${r.year}</td>
        <td>${r.model.toLocaleString()}</td>
        <td>${r.rosstat?.toLocaleString()||'-'}</td>
        <td>${r.diff?.toLocaleString()||'-'}</td>
        <td>${dp}</td>
      </tr>`;
    });
    html += '</tbody></table>';
    document.getElementById('compareTableContainer').innerHTML = html;
  } catch(e) {
    alert('Ошибка прогноза: '+e);
  }
};
