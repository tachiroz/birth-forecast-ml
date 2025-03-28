// Функция для отображения параметров в зависимости от выбранной модели
document.getElementById('modelSelect').addEventListener('change', function() {
    const model = this.value;
    const parametersDiv = document.getElementById('parameters');
    parametersDiv.innerHTML = ''; // Очистка предыдущих параметров
  
    if (model === 'linear_regression') {
      parametersDiv.innerHTML += '<label>Для линейной регрессии параметры не требуются</label>';
    } else if (model === 'decision_tree') {
      parametersDiv.innerHTML += '<label for="maxDepth">Максимальная глубина дерева</label>';
      parametersDiv.innerHTML += '<input type="number" id="maxDepth" placeholder="Введите максимальную глубину">';
    } else if (model === 'random_forest') {
      parametersDiv.innerHTML += '<label for="numTrees">Количество деревьев</label>';
      parametersDiv.innerHTML += '<input type="number" id="numTrees" placeholder="Введите количество деревьев">';
      parametersDiv.innerHTML += '<label for="maxDepthRF">Максимальная глубина</label>';
      parametersDiv.innerHTML += '<input type="number" id="maxDepthRF" placeholder="Введите максимальную глубину">';
    } else if (model === 'neural_network') {
      parametersDiv.innerHTML += '<label for="layers">Количество слоев</label>';
      parametersDiv.innerHTML += '<input type="number" id="layers" placeholder="Введите количество слоев">';
      parametersDiv.innerHTML += '<label for="neurons">Количество нейронов в слое</label>';
      parametersDiv.innerHTML += '<input type="number" id="neurons" placeholder="Введите количество нейронов">';
      parametersDiv.innerHTML += '<label for="learningRate">Скорость обучения</label>';
      parametersDiv.innerHTML += '<input type="number" id="learningRate" step="0.001" placeholder="Введите скорость обучения">';
    }
  });
  
  // Обработчик для кнопки "Обучить"
  document.getElementById('trainButton').addEventListener('click', function() {
    // Здесь можно добавить логику для передачи данных на backend
    alert('Начало обучения модели!');
  });
  