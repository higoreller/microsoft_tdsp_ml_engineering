## Projeto básico de uma rest API de machine learning utilizando o padrão TDSP da Microsoft

# Contexto

Neste código, dois modelos diferentes de aprendizado de máquina são treinados, ajustados e registrados usando o MLflow: um modelo de regressão logística e um modelo de floresta aleatória. O código importa as bibliotecas necessárias, define funções para carregar e pré-processar o conjunto de dados e cria um invólucro PythonModel para os modelos. O conjunto de dados é dividido em conjuntos de treinamento e teste, e o PyCaret é usado para configurar o ambiente e criar e ajustar os modelos. Os modelos e suas métricas são então registrados usando o MLflow.

Após o treinamento, um novo conjunto de dados é pré-processado e enviado como uma solicitação POST para uma API. A API retorna previsões para os novos dados e a perda de log e a pontuação f1 são calculadas e impressas.
