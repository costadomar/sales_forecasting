# Previsão da Quantidade de Vendas com Série Temporais

![log_series_temporais](https://user-images.githubusercontent.com/90925360/187972304-0853346c-51e6-4ee6-841e-5940f2d9f091.JPG)
  
## Descrição da Base de Dados:
É composta por dados históricos diários de vendas, esta disponivel aqui: <https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data> 
### Descrição das variaveis:
* ID  - um ID que representa uma tupla (Loja, Item) dentro do conjunto de teste
* shop_id - identificador exclusivo de uma loja
* item_id - identificador exclusivo de um produto
* item_category_id - identificador exclusivo da categoria do item
* item_cnt_day - número de produtos vendidos. Você está prevendo um valor mensal dessa medida
* item_price - preço atual de um item
* data  - data no formato dd/mm/aaaa
* date_block_num - um número de mês consecutivo, usado por conveniência. Janeiro de 2013 é 0, fevereiro de 2013 é 1,..., outubro de 2015 é 33
* item_name  - nome do item
* shop_name - nome da loja
* item_category_name - nome da categoria do item

## Objetivo: 
Criar uma análise exploratória e desenvolver um modelo de
forecast de vendas para um dataset sobre venda de produtos em uma empresa
de varejo. O modelo deve prever as vendas para um só produto de apenas uma loja
alvo no próximo mês.

## Abordagem do Problema
A abordagem utilizada aqui vai ser de Séries Temporais, devido um dos objetivos de uma série temporal ser é a compreensão dessa serie ao londo do tempo, sendo de suma importância, quando falamos, por exemplo, de uma análise de crescimento ou decrescimento de vendas que é o caso estudado aqui, para poder tomar a mehor decisão partindo desse tipo de análise. Com o estudo dela, podemos realizar a previsão futura da venda do próximo mês.
## Entendendo os dados:

Fazendo o import das bibliotecas necessária
```
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from statsmodels.tsa.api import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error
from pmdarima.metrics import smape

import warnings
warnings.filterwarnings('ignore')
```
Para a análise aqui nesse projeto, só vamos utilizar o dataset < sales_train >,devido conter todas as inforrmações necessárias com o objetivo do projeto:

![image](https://user-images.githubusercontent.com/90925360/187980485-1e98ddf3-2e38-4e34-ae49-c8053eed37ad.png)

