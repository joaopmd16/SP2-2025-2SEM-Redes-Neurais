# Sprint Challenge - Redes Neurais (LSTM para gravidade de acidentes)
# Alunos: Pietro RM562407, João RM561738, Nelson RM565603, Vitor RM566181
# Link para o colab: https://colab.research.google.com/drive/1faLgxyrPhei3-8JCgCu8C0hggI_PhIHg?usp=sharing
# Link para o drive: https://drive.google.com/drive/folders/1IqwEnIFFWFtfy8R23qQCMUEgUHuasse_?usp=sharing
# Link para o loom: https://www.loom.com/share/74f0f36e88c74332a0274dc36dcedc8a

Este projeto implementa uma rede neural recorrente (LSTM) para prever se um acidente é **severo ou não severo**, com base em dados históricos de acidentes agregados mensalmente.

# Estrutura do Projeto

- `SPRINTCHALLENGE_REDES_NEURAIS_final.ipynb` — Notebook principal com todo o código.
- `datatran2025.csv` — Base de dados com registros de acidentes (usada no Google Drive).
- `relatorio_final_SPRINT_REDES.pdf` — Relatório explicando as decisões e arquitetura do modelo.

# Execução

# Pré-requisitos
- Python 3.10+
- Google Colab (ou Jupyter Notebook)
- Pacotes principais:
  ```bash
  pip install tensorflow scikit-learn pandas numpy matplotlib
  ```

# Como executar

1. Monte seu Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Defina o caminho da pasta:
   ```python
   pasta = '/content/drive/MyDrive/REDES_CHAL_2S2/'
   file_path = pasta + 'datatran2025.csv'
   ```

3. Execute todas as células do notebook `SPRINTCHALLENGE_REDES_NEURAIS_final.ipynb`.

4. O modelo exibirá:
   - Gráficos de perda (treino/validação)
   - Relatórios de classificação (`classification_report`)
   - Matriz de confusão
   - ROC-AUC Score

# Modelo

A arquitetura da rede LSTM inclui:
- LSTM(64, return_sequences=True)
- Dropout(0.2)
- LSTM(32)
- Dense(1, activation='sigmoid')

# Otimizador e Métricas
- Otimizador: `Adam`
- Função de perda: `Binary Crossentropy`
- Métrica: `Accuracy`

# Métricas de Avaliação
- ROC-AUC
- Matriz de confusão
- Precision, Recall e F1-score
