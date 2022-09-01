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
###1. Importando as Bibliotecas
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

###2. Análise Descritiva dos Dados:

![image](https://user-images.githubusercontent.com/90925360/187981278-84ed7224-8c2a-4a39-988d-ebe18fd646ad.png)

![image](https://user-images.githubusercontent.com/90925360/187982013-e203710a-50f8-4bfe-9476-d28721be14e2.png)

 Como a base de dados inicial apresenta só 6 dados duplicados, vamos dropá-los do dataset para uma melhor análise.
 ```
df_sales_train.drop_duplicates(inplace=True) 
```
Como estamos trabalhando com série temporal, temos que organizar de forma crescente a nossa base de acordo com o tempo. Primeiramente, tranformar o type da variavel <date> em <datetime>, e logo em seguida fazer esse ordenamento:
   ```
df_sales_train  = df_sales_train.sort_values(by= 'date', axis=0, ascending=True).reset_index(drop=True)
```
###3. Análise Descritiva dos Dados:
  Aqui temos um overview da distribuição e do box-plot das variaveis do dateset.
  
  ![image](https://user-images.githubusercontent.com/90925360/187983548-80a0e78f-6893-48aa-8b0a-21d70f723bc9.png)
  
  ![image](https://user-images.githubusercontent.com/90925360/187983657-cd627bc9-a87b-4339-ad39-c59ac089425b.png)
  
Os dados do histograma acima mostram a distribuição dos dados. Os pontos importantes, é que a variavel <shop_id> correspondentes as lojas, mostrou que temos mais frequencia da loja 31. Já as variaveis <Item_price> e <item_cnt_day> indicaram pontos de outliers, como mostra os box-plot abaixo. Não, vamos fazer o tratamento de outliers ainda , por que esse ainda não é o dataset final que irá ser trabalhado para o modelo.
  
![image](https://user-images.githubusercontent.com/90925360/187984453-e3801d0d-9bf0-47a1-8bf7-1eaf6702b507.png)
  
 O gráfico nos mostra uma leve tendencia de queda com o passar do tempo, principalmente a partir de 2015.

###3.1 Visualizando os dados por Produto
Vamos fazer um filtro para verificar qual o produto que mais tem saida.
  
 ```
item_top10 = pd.DataFrame(df_sales_train.groupby('item_id')['item_cnt_day'].sum()\
                        .rename({'item_cnt_day': 'quantidade'}, axis='columns'))\
                        .sort_values(by = ['item_cnt_day'], ascending = False )\
                            .reset_index().head(10)
```
  
Aqui para uma melhor visualização, filtrei só o top 10 dos produtos.

![image](https://user-images.githubusercontent.com/90925360/187985036-e9c6a3ac-ba11-4b96-b53c-64cc67062675.png)
  
No gráfico acima mostra o top 10 de produtos por quantidade de vendas. Por isso, o produto escolhido para análise vai ser o **20949** que apresenta uma saida de produtos perto dos 188k.

A partir disso, vamos filtrar no nossa base o produto escolhido e continuar a análise dos dados.
