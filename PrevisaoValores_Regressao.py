import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 📌 Carregar os dados do arquivo CSV
data = pd.read_csv("data.csv")

# 📌 Verificar os primeiros registros
print(data.head())

# 📌 Converter a coluna 'RM' para inteiro
data['RM'] = data['RM'].astype(int)

# 📌 Criar categorias com base no número de quartos
categorias = []
for i in data.RM.items():
    valor = i[1]
    if valor <= 4:
        categorias.append('Pequeno')
    elif valor < 7:
        categorias.append('Medio')
    else:
        categorias.append('Grande')

data['categorias'] = categorias

# 📌 Média de preços por categoria
medias_categorias = data.groupby('categorias')['MEDV'].mean()
dic_baseline = {
    'Grande': medias_categorias.get('Grande', 0),
    'Medio': medias_categorias.get('Medio', 0),
    'Pequeno': medias_categorias.get('Pequeno', 0)
}

# 📌 Função para prever valores com baseline
def retorna_baseline(num_quartos):
    if num_quartos <= 4:
        return dic_baseline.get('Pequeno', 0)
    elif num_quartos < 7:
        return dic_baseline.get('Medio', 0)
    else:
        return dic_baseline.get('Grande', 0)

# 📌 Criar conjuntos de treino e teste
X = data.drop(columns=[col for col in ['RAD', 'TAX', 'MEDV', 'DIS', 'AGE', 'ZN', 'categorias'] if col in data.columns])

y = data['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# 📌 Criar baseline
predicoes_baseline = [retorna_baseline(valor) for _, valor in X_test.RM.items()]



# 📌 Criar DataFrame para comparar resultados
df_results = pd.DataFrame({
    'valor_real': y_test.values,
    'valor_predito_baseline': predicoes_baseline
})

# 📌 Calcular erro RMSE do baseline
rmse_baseline = np.sqrt(mean_squared_error(y_test, predicoes_baseline))
print(f'RMSE do Baseline: {rmse_baseline:.2f}')

# 📌 Treinar Regressão Linear
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
df_results['valor_predito_reg_linear'] = lin_model.predict(X_test)

# 📌 Treinar Decision Tree
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
df_results['valor_predito_arvore'] = tree_model.predict(X_test)

# 📌 Treinar Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
df_results['valor_predito_random_forest'] = rf_model.predict(X_test)

# 📌 Calcular RMSE para cada modelo
rmse_linear = np.sqrt(mean_squared_error(y_test, df_results['valor_predito_reg_linear']))
rmse_tree = np.sqrt(mean_squared_error(y_test, df_results['valor_predito_arvore']))
rmse_rf = np.sqrt(mean_squared_error(y_test, df_results['valor_predito_random_forest']))

print(f'RMSE Regressão Linear: {rmse_linear:.2f}')
print(f'RMSE Decision Tree: {rmse_tree:.2f}')
print(f'RMSE Random Forest: {rmse_rf:.2f}')

# 📌 Criar gráfico para comparar os modelos
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_results.index, y=df_results.valor_real, mode='markers', name='Valor Real'))
fig.add_trace(go.Scatter(x=df_results.index, y=df_results.valor_predito_baseline, mode='lines+markers', name='Baseline'))
fig.add_trace(go.Scatter(x=df_results.index, y=df_results.valor_predito_reg_linear, mode='lines', name='Regressão Linear'))
fig.add_trace(go.Scatter(x=df_results.index, y=df_results.valor_predito_arvore, mode='lines', name='Árvore de Decisão'))
fig.add_trace(go.Scatter(x=df_results.index, y=df_results.valor_predito_random_forest, mode='lines', name='Random Forest'))

# 📌 Exibir gráfico interativo
fig.show()

# 📌 Salvar dados atualizados em CSV
data.to_csv('data_atualizado.csv', index=False)
print("Arquivo 'data_atualizado.csv' salvo com sucesso!")

