# Project Charter

## Business background

- O cliente pode ser qualquer pessoa interessada na previsões de jogadas. Pode ser para obter as melhores posições e tipos de arremessos
- Estamos tentando abordar o problema de prever a probabilidade de acerto dos arremessos durante os jogos.

## Scope

- Estamos construindo uma solução de ciência de dados para prever a probabilidade de acerto de arremessos no basquete.
- Desenvolveremos um modelo baseado em dados de arremessos feitos pelo astro da NBA, Kobe Bryant, durante sua carreira.
- O modelo será consumido pelo cliente por meio de uma API.

## Personnel

- Quem está envolvido no projeto: o Cientista de dados e Engenheiro de machine learning.

## Metrics

- Modelo de Regressão Logística (primeiro bloco de resultados): log_loss
- Modelo de Floresta Aleatória (segundo bloco de resultados): f1_score e log_loss
- Novos dados, modelo de Regressão Logística: f1_score e log_loss
- Outras métricas foram obtidas, mas não foram alvo de análise
