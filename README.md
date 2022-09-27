# Previsão da Quantidade de Vendas com Séries Temporais

![log_series_temporais](https://user-images.githubusercontent.com/90925360/187972304-0853346c-51e6-4ee6-841e-5940f2d9f091.JPG)
  
## Descrição da Base de Dados:
É composta por dados históricos diários de vendas, está disponivel aqui: <https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data> 
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
A abordagem utilizada aqui vai ser de Séries Temporais, devido um dos objetivos de uma série temporal ser é a compreensão dessa serie ao londo do tempo, sendo de suma importância, quando falamos, por exemplo, de uma análise de crescimento ou decrescimento de vendas que é o caso estudado aqui, para poder tomar a mehor decisão partindo desse tipo de análise. Com o estudo dela, podemos realizar a previsão futura da venda do próximo mês. Com isso, os modelos que irão ser empregados aqui vai ser o ARIMA e o SARIMAX.
## Entendendo os dados:
## 1. Importando as Bibliotecas
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

## 2. Análise Descritiva dos Dados:

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
## 3. Análise Descritiva dos Dados:
  
  Aqui temos um overview da distribuição e do box-plot das variaveis do dateset.
  
  ![image](https://user-images.githubusercontent.com/90925360/187983548-80a0e78f-6893-48aa-8b0a-21d70f723bc9.png)
  
  ![image](https://user-images.githubusercontent.com/90925360/187983657-cd627bc9-a87b-4339-ad39-c59ac089425b.png)
  
Os dados do histograma acima mostram a distribuição dos dados. Os pontos importantes, é que a variavel <shop_id> correspondentes as lojas, mostrou que temos mais frequencia da loja 31. Já as variaveis <Item_price> e <item_cnt_day> indicaram pontos de outliers, como mostra os box-plot abaixo. Não, vamos fazer o tratamento de outliers ainda , por que esse ainda não é o dataset final que irá ser trabalhado para o modelo.
  
![image](https://user-images.githubusercontent.com/90925360/187984453-e3801d0d-9bf0-47a1-8bf7-1eaf6702b507.png)
  
 O gráfico nos mostra uma leve tendencia de queda com o passar do tempo, principalmente a partir de 2015.

### 3.1 Visualizando os dados por Produto
  
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
  
   ```
Item_20949 = df_sales_train[df_sales_train['item_id'] == 20949].reset_index(drop= True)
```
  
 Visualizando a quantidade de vendas diária e mensais só do produto escolhido.
  
  ![image](https://user-images.githubusercontent.com/90925360/187986600-3c7b02bf-ed32-49d5-a898-5b5563efcebb.png)

  ![image](https://user-images.githubusercontent.com/90925360/187986388-bc287f28-4419-48b2-94d8-d4b45fe77452.png)
  
Os dos gráfico acima, mostram picos de vendas em janeiro de 2014 e 2015 e uma leve tendencia de queda na quantidade de vendas a partir de 2015.
  
Com isso, agora falta escolher a loja para para seguir para a modelagem da nossa série.

### 3.2 Visualizando os dados por loja
  
O critério de escolha da loja, vai ser a loja que tem mais saida do produto escolhido (20949). Por isso, vamos filtrar a loja por quantidade.
  
   ```
loja_top10 = pd.DataFrame(Item_20949.groupby('shop_id')['item_cnt_day'].sum()\
    .rename({'item_cnt_day': 'quantidade'}, axis='columns'))\
    .sort_values(by='item_cnt_day', ascending=False)\
    .reset_index().head(10)
loja_top10
```
Vamos, visualizar o top 10 das lojas.
 
  ![image](https://user-images.githubusercontent.com/90925360/187987677-0c2e5f5f-6ee8-44f7-a5f3-cd157787832f.png)
  
A Loja 31 é que apresenta a maior quantidade desse produto em vendas. Então, é por esse motivo que vai ser a loja escolhida para a nossa análise.
  
### 3.3 Visualização da loja 31 pelo produto 20949
  
Agora já escolhemos qual o produto e loja que vamos utilizar para análise. Com isso, vamos fazer mais algumas visualização dos dados, para um melhor entendimento da base final.
 
```
df_shop31_Item_20949= Item_20949[Item_20949['shop_id'] == 31].reset_index(drop = True)
```
### 3.2.1 Análise Descritiva dos Dados

A variavel alvo a ser estudada aqui na análise é <item_cnt_day>, na qual nos mostra a quantidade de vendas dos produtos, por isso vamos plotar os gráficos agora em função dela.
  
![image](https://user-images.githubusercontent.com/90925360/187989177-63d9953f-e578-4905-88b1-e99f4af5c05d.png)
  
![image](https://user-images.githubusercontent.com/90925360/187989529-0931e69a-44ec-4ade-8733-20f21f9480f2.png)
  
Como podemos observar no histograma, ele apresenta um distribuição assimétrica a direita, com alguns outliers de quantidades de vendas a partir de 60 em concordância com o gráfico de box-plot abaixo que nos mostra que 50% da quantidade dos produtos está um pouco acima de 20.
 
![image](https://user-images.githubusercontent.com/90925360/187989954-af855620-87b1-4ea8-a297-32062e6687a9.png)
  
![image](https://user-images.githubusercontent.com/90925360/187990121-fb1a8ba9-3ac1-440a-bad6-0c48b21b3bc7.png)

Os gráficos acima, apresentam uma a série temporal da nossa variavel target por dia e por mês ao longo do tempo.
  
### 3.2.1 Análise de Correlação
  
Vamos fazer uma análise de correlação da variável target em função do tempo para analisar o comportamento.

![image](https://user-images.githubusercontent.com/90925360/187990633-4d554454-f151-4381-9ac5-fa4184c6d4a0.png)

 A série não apresenta uma correlação alta com o tempo.

## 4. Análise Da Série Temporal 'item_cnt_day'
  
A partir daqui vamos analisar o comportamento da nossa variável target, mediante conceitos de séries temporais. Como queremos prever o próximo mês de venda, vamos trabalhar com o dataset organizado mensalmente.

 ```
df_mensal = df_shop31_Item_20949.copy() #fazendo uma cópia do dataset
df_mensal.set_index('date', inplace = True) #transformando a coluna data em index
df_mensal = df_mensal.resample(rule = 'M').last() #reordenando o dataset por mês
```
 
### 4.1 Decomposição dos dados
 As séries temporais apresentam algumas propriedades importantes para serem analisas:
  * Tendência: É a análise se a série esta crescendo, dimuindo ou estável com o decorrer do tempo.
  * Sazonalidade: Seria um feômeno periodo que se repetem no mesmo periodo no tempo, exemplo, vendas de chocolate em toda pascoa no mês de março pr exemplo.
  
![image](https://user-images.githubusercontent.com/90925360/188003933-77bc6dee-bf05-456a-8315-6602ae682feb.png)

Claramente a série apresenta uma tendência de baixa a partir de maio de 2014. Apresentando também picos no começo de cada ano, indicando alguma sazonalidade.
  
### 4.2 Autocorrelação e Autocorrelação Parcial
  
 A função de autocorrelaçao mede o quanto a série esta relacionada com os seus valores passados, no nosso caso, quanto o nossa quantidade mensal de venda esta relacionadas os valores dos meses que já passaram. Na autocorrelação parcial, é medido a correlação entre duas observações seriais, ou seja, dois periodos diferentes.
  
 ![image](https://user-images.githubusercontent.com/90925360/188005017-0a9013c9-8d11-4b13-9ddc-7694e11fd2c8.png)
 
  Como podemos observar, não temos nenhum lag importante que seja correlacionado com a nossa série.
  
 ### 4.3 Estacionariedade da Série
  
 Aqui vamos checar a estacionariedade da série. Para ser estacionária a série deve apresentar média,variância e a estrutura de autocorrelaçao se mantém constantes durante o tempo.
  
![image](https://user-images.githubusercontent.com/90925360/188008037-b278a5e9-9e6c-4de2-a1e4-ac400f40833d.png) 
  
 A imagem apresenta uma janela de tempo de 4 meses, para verificar o comportamento da média. Como podemos verificar a média apresenta dois picos ali, como já comentados em outras análises.

  Para uma melhor entendimento se a série é estacionária, vamos realizar o teste estatístico de Dickey-Fuller. O teste nada mais é que um teste de hipótese:
  
```
  ADF_test(df_mensal['item_cnt_day'],'quantidade_produto')
```
  
  ![image](https://user-images.githubusercontent.com/90925360/188009191-9b6a4da6-1a60-4fc2-9fc9-1f86f07448a3.png)

O teste de dick fuller, nos mostrou que a série é estacionária com uma confiança de 90%,95% e 99%, mesmo mostrando aqueles picos vistos na sazonalidade e aquela pequena tendência de queda.
  
### 4.3.1 Fazendo a diferenciação (DIFF)
  
 O método diff tem como finalidade tirar a sazonalidade da série, para o modelo de série temporal ter uma melhor captura do comportamento da série analisada aqui. Nesse caso, vamo tirar a diferença de 1 mês.
 
  ```
def get_diff(data):
    data['item_cnt_day_diff'] = data.item_cnt_day.diff()    
    data = data.dropna()      
    return data
stationary_df = get_diff(df_mensal)
```
 
![image](https://user-images.githubusercontent.com/90925360/188009969-e5d7a2bc-7bbc-4a0d-aa7d-27de29fd72e5.png)

 Vamos, analisar novamente a média movel e a estacionariedade da série, mas agora com a diferenciaçao.
  
![image](https://user-images.githubusercontent.com/90925360/188010082-e4a0853f-1dd5-44a6-8d65-62db0ee391a9.png)

![image](https://user-images.githubusercontent.com/90925360/188010453-19f564a0-f5c7-4cdb-bc25-2b0acef43183.png)
  
A nossa série continua sendo estacionária, mas veja que conseguimos deixar a média (Figura acima) um pouco melhor para o modelo. Por isso, a modelagem vai ser feita na variável com diferenciação.

## 5. Engenharia de Features 
  
Vamos criar algumas features para tentar melhorar a previsão do modelo.
  
```
stationary_df['mes'] = [i.month for i in stationary_df.index] #add uma coluna de mês
stationary_df['ano'] = [i.year for i in stationary_df.index] #add uma couna de ano

```
Com isso, temos o nosso dateset que irá seguir para o modelo.

  ![image](https://user-images.githubusercontent.com/90925360/188011292-48f3a32c-9a7f-468f-a857-964db241338c.png)

## 6. Separação da Base de Treino e Teste

Vamos retirar da base variáveis que não faz sentido para o modelo, como: o número da loja, o número do produto, o valor do produto, devido ser variáveis que possuem o mesmo valor ao longo de todo o dataset, como também a variável <date_block_num> já que temos uma coluna de mês.
  
```
x=stationary_df.drop(columns =['item_cnt_day', 'sales_month', 'shop_id', 'item_id', 'date_block_num', 'item_price', 'item_cnt_day_diff'])
y=stationary_df.item_cnt_day_diff
x = sm.add_constant(x) #addicionando uma constante base da variavel regressoras ou exogenas.
```

Com isso, as variáveis regressoras ou exógenas ficaram assim:
 
![image](https://user-images.githubusercontent.com/90925360/188012434-5b39c2cb-64e2-4dd4-80b9-6dab7be58577.png)

 Agora, vamos dividir o y e o x na nossa base de treino e teste. Como estamos falando de série temporal o tempo deve ser respeitado, então, separei a minha base de treino num período que vai 2013-01-31 a 2015-07-31 e base de teste os últimos 5 meses.
  
```
x_train, x_test, y_train, y_test = train_test_split(x,y, '2013-01-31','2015-07-31', '2015-08-31') 
``` 
![image](https://user-images.githubusercontent.com/90925360/188013043-75dc193a-0c4e-4f0b-9168-24f6e489d59d.png)

## 7. Escalonando as variáveis
  
```
scaler = MinMaxScaler()
scaler.fit(x_train)

X_train_scaled = scaler.transform(x_train)
X_test_scaled = scaler.transform(x_test)
X_escalo = scaler.transform(x)
# Transformand para dataframe para visualização
x_train = pd.DataFrame(X_train_scaled,columns = x_train.columns, index =x_train.index )
x_test = pd.DataFrame(X_test_scaled,columns = x_train.columns, index = x_test.index)
x = pd.DataFrame(X_escalo,columns = x.columns, index = x.index)
```

## 8. Modelando a Série Temporal

Para a modelagem da série temporal, foi escohido dois tipos de modelos o ARIMA e a sua derivação o SARIMAX. São modelos estatísticos muito utlizados para esse tipo de abordagem.

### 8.1 Modelando com o Auto-arima

Para analisarmos mais profundamente os dados que nos foi fornecido, utilizamos o AUTO ARIMA. Sendo assim, escolhemos iniciar os parâmetros p e q (autorregressivo e média móvel, respectivamente) em 0 e o m (periodicidade de sazonalidade) em 12 devido a nossa base ser em meses. Com isso, ele irá testar várias configurações de modelos ARIMA e trazer o melhor que minimiza o AIC.

![image](https://user-images.githubusercontent.com/90925360/188015504-31c08a13-bab2-4e55-82ee-587e0a6a6094.png)

![image](https://user-images.githubusercontent.com/90925360/188015605-892d9a0a-55a8-4210-9b29-0516c64d99e9.png)

Esses são os valores dos parâmetros que minimizam melhor o AIC.

Com isso, vamos treinar o modelo a partir desses parâmetros, fazendo uma previsão de 5 meses a frente da nossa base de treino.

![image](https://user-images.githubusercontent.com/90925360/188016237-4258ccf8-643d-476b-b464-34d1f153bb61.png)

![image](https://user-images.githubusercontent.com/90925360/188016301-0a74dac2-ea36-480b-a4fd-e88978ce1c01.png)

Podemos vê a previsão do modelo (verde) em relação aos dados de testes. A predição até consegue fazer a tendência nos 2 primeiros meses do dados de teste, mas logo em seguida ela começa a subir.
  
Vamos analisar os resíduos gerados pelo modelo.

```
  stepwise.plot_diagnostics(figsize=(16, 8))
  plt.show()
```
![image](https://user-images.githubusercontent.com/90925360/188018189-e4185ac8-6cab-447f-a96d-c2312020a213.png)

Na figura acima, temos o resíduo, histograma dos resíduos, o gráfico QQ e o correlograma.

O comportamento dos resíduos apresentam uma grande flutuação em torno do valor zero. Já o histograma não chegou a apresentar uma distribuição normal. No gráfico, do Q-Q plot temos uma grande dispersão dos valões no começo da linha vermelha e no final dela, o que evidencia uma não normalidade dos dados, e por fim, o correlograma que não traz muitas informações.

Agora vamos analisar as métricas do nosso modelo. 

As métricas para a avaliação do modelo é o RMSE, MAE da biblioteca SCIKIT-LEARN, sendo muito usadas em séries temporais A primeira métrica é a Raiz Quadrada do Erro Quadrático Médio, sendo a diferença do valor real e o valor previsto, é usada para comparação entre modelos, onde o melhor modelo apresenta o valor mais próximo de zero, sendo mais sensível a erros maiores O MAE, erro médio absoluto, sendo a média dos erros absolutos da série, na qual é a diferença do real com o previsto.

![image](https://user-images.githubusercontent.com/90925360/188019943-dbe8d18d-ef0a-433f-84e1-7e559cc6fd60.png)

### 8.2 Modelando com o Sarimax

Para seguir a modelagem com o SARIMAX, utilizaremos os coeficientes dados pelo AUTO-ARIMA, order (0,0,1) e seasonal_order (0,0,1,12). Primeiramente vamos fazer a modelagem só com a variavel target, sem as variaveis exogenas no modelo para analizarmos o comportamento do modelo.

```
  mod = SARIMAX(y_train, order=(0,0,1), seasonal_order=(0, 0, 1, 12))
  fit_res = mod.fit(disp=False, maxiter=250)
  print(fit_res.summary())

  predito = fit_res.predict(typ='levels')
```
![image](https://user-images.githubusercontent.com/90925360/188023000-22ea990c-d400-46f7-957b-e191ee1c8ab3.png)

Quanto mais próximo o grau de conformidade e previsão as linhas devem ficar alinhadas sem recuos. Como podemos ver acima, o modelo apresentado conseguiu capturar um pouco o padrão das bases de dados de treino.
 
 Agora, vamos analisar a perfomance na série com a base de teste.
  
 ![image](https://user-images.githubusercontent.com/90925360/188028199-58186f94-2f41-4a83-9cfc-f3b08cfcc3f2.png)

Aqui, temos a Previsão do Sarimax em relação a nossa base de dados. Vemos que o modelo conseguiu acompanhar a tendência dos dados ao longo da série.

Vamos analisar os resíduos e as métricas para vê o desempenho do modelo.

 ![image](https://user-images.githubusercontent.com/90925360/188028327-1a469a7c-8e4e-48a3-9b20-de68a78bd27e.png)

 ![image](https://user-images.githubusercontent.com/90925360/188028441-9f8b5385-f908-4d46-b8c8-b38c32b3ecdc.png)

Como podemos perceber o SARIMAX obteve uma melhora em relação ao modelo do ARIMA. A análise de resíduos, ainda mostrou uma dispersão no começo e no final, mas a maioria dos valões permanceram em cima da linha vermelha, o histograma já apresentou uma melhor distribuição se aproximando da normal, como mostra a linha do KDE. Em relação, as métricas tivemos uma pequena diferença também de perfomance.
Por conta disso, vamos seguir a modelagem com o Sarimax.

### 8.2.1 Agregando Variaveis Exogenas ao Modelo
 Agora, vamos agregar as variáveis exógenas ao nosso modelo para vê se a perfomance dele melhora ou piora.
  
 ```
mod = SARIMAX(y_train,exog=x_train, order=(0,0,1), seasonal_order=(0, 0, 1, 12))
fit_res = mod.fit(disp=False, maxiter=250)
print(fit_res.summary())

predito = fit_res.predict(typ='levels')
  ```
  ![image](https://user-images.githubusercontent.com/90925360/188030472-f6d6a64a-956b-4632-a2e8-5256314833f2.png)

  ![image](https://user-images.githubusercontent.com/90925360/188032056-58fda351-6fb0-44b0-8087-5341a553becf.png)

Aqui temos, na primeira figura o modelo só com a base de treino e na figura debaixo o modelo com toda a base de dados. Aparentemente, a perfomance foi parecida com o modelo sem as variáveis exógenas.
  
Seguimos com a análise dos residuos e das métricas.
 
 ![image](https://user-images.githubusercontent.com/90925360/188032363-6cbea940-9860-4dd1-bfc3-933d6ff017d6.png)
  
 ![image](https://user-images.githubusercontent.com/90925360/188032432-fd5bbd5c-af06-48f0-9ae7-6636f9dc6389.png)
  
Podemos verificar pela análise de residuos, que obteve uma perfomance parecida, mas vale destacar Q-Q plot apresenta mais valões em cima da linha vermelha, principalmente no inicio da série, na qual podemos verificar que o modelo conseguiu capturar melhor o padrão da série quando comparados ao modelo sem variaveis exogenas e o arima. Na qual, comprova-se com os valores das métricas que obtiveram um melhor resultado, tanto mean absolute error, quanto o rmse.

### 8.2.2 Usando o Cross-Validation
Para entendermos melhor o comportamento do modelo ao longo da série temporal aqui estudada, vamos utilizar um cross-validation, respeitando o conceito de séries temporais. A base de treino ira iniciar com 5 meses e conforme o passar da função 1 mês vai ser adicionado a base de treino, já a de teste sempre será 5 meses a frente da base de treino.

 ```
rmse_treino =[]
rmse_teste = []

mae_treino =[]
mae_test = []
cont = 0
for i in range(5, df_mensal.shape[0] - 4):
  cont += 1
  x_train =x.iloc[:i]
  x_test = x.iloc[x_train.shape[0]:x_train.shape[0]+5]
  y_train =y.iloc[:i]
  y_test = y.iloc[x_train.shape[0]:x_train.shape[0]+5]
  x_t = x.iloc[:x_train.shape[0]+5]
  y_t= y.iloc[:x_train.shape[0]+5]
    

  modelo3 = SARIMAX(y_train, exog = x_train, order=(3,0,0), seasonal_order=(1, 0, [1, 2], 12), enforce_stationarity=False)
  
  fit_res = modelo3.fit(disp=False, maxiter=250)
  predito = fit_res.predict(typ='levels')
  rmse_treino.append(np.sqrt(mean_squared_error(y_train, predito)))
  mae_treino.append(mean_absolute_error(y_train, predito))

  modelo3 = SARIMAX(y_t, exog = x_t, order=(3,0,0), seasonal_order=(1, 0, [1, 2], 12),enforce_stationarity=False )

  res1 = modelo3.filter(fit_res.params)

  predict = res1.get_prediction()
  predict_ci = predict.conf_int()

  
  # Predições dinâmicas
  predict_dy = res1.get_prediction(dynamic= y_test.index[0])
  predict_dy_ci = predict_dy.conf_int()
  rmse_teste.append(np.sqrt(mean_squared_error(y_test, predict_dy.predicted_mean.loc[y_test.index[0]:])))
  mae_test.append(mean_absolute_error(y_test, predict_dy.predicted_mean.loc[y_test.index[0]:]))

  print(f"RMSE {cont}: { mean_squared_error(y_train, predito, squared= False):.2f}")
  print(f"MAE {cont}: {mean_absolute_error(y_train, predito)}")

print()
med_erro = np.mean(mae_test)  
print(f"RMSE_média_treino: {np.mean(rmse_treino):.2f}")
print(f"MAE_med_treino: {np.mean(mae_treino):.2f}")
print(f"RMSE_média_test: {np.mean(rmse_teste):.2f}")
print(f"MAE_med_test: {np.mean(mae_test):.2f}")
 ```
  
O resultados das métricas ao longo da série:
  
![image](https://user-images.githubusercontent.com/90925360/188034529-e60c03b1-d9ba-438d-a637-2412b7b70203.png)
  
![image](https://user-images.githubusercontent.com/90925360/188035152-5790aa5b-2f9d-4bb8-ac76-a2085df9b089.png)

 ![image](https://user-images.githubusercontent.com/90925360/188034621-fceb7e0f-8697-41a5-a0b3-7476f059444d.png)

Temos o comportamento das métricas ao longo do tempo. As duas métricas tem um pico de erro, no começo de 2014 e logo em seguida no começo de 2015, se formos analisar pelo comportamento da nossa série, ela apresenta pico nos começo desses anos, então se torna aceitavel que o modelo erre quando ele tenta fazer previsões nessas datas. Mas, tanto no RMSE quanto MAE na base de teste, temos um tendência de queda ao longo da série.

A média do MAE pelo modelo foi de 19.93 e a do RMSE de 24.
  
Com isso, vamos partir para o forecast com o esse modelo treinado.

## 9. Forecasting da Quantidade de Vendas

O gráfico e a tabela abaixo apresentam a previsão futura de 5 meses a frente, como o modelo SARIMAX  com a inclusão das variáveis exógenas.
  
![image](https://user-images.githubusercontent.com/90925360/188037580-6dcdf8c8-5f4d-4eb8-b41f-5dd5d51fbb0a.png)

Apresenta a média das previsões, juntamente com o intervalo de confiança de 95%.
  
Para um melhor diagnóstico para o negócio, apresento também uma tabela com o melhor e o pior cenário de vendas, em relação ao erro médio do modelo (MAE):

```
pred_media = pred_uc.predicted_mean.reset_index()
pred_media.columns = ['data_previsão','Media_predicao']
pred_media['Pior_Cenário'] = pred_media['Media_predicao'].map(lambda x: x - med_erro)
pred_media['Melhor_Cenário'] = pred_media['Media_predicao'].map(lambda x: x + med_erro)
```

![image](https://user-images.githubusercontent.com/90925360/188038357-b9d9d4bb-d012-4b65-9d61-7dbbbe0cac26.png)

## 10. Considerações Finais

O modelo ainda pode ser melhorado, com a adição de algumas features para uma melhor perfomance.
