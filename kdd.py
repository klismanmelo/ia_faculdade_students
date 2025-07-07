from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

file_path = "dataset.csv"
df = pd.read_csv(file_path)

# Exibir as primeiras linhas e os tipos de dados para análise inicial
df.head(), df.dtypes

# Etapa 1: Seleção dos dados relevantes
data = df.drop(columns=["student_id"])

# Etapa 2: Pré-processamento - remover valores ausentes
data = data.dropna()

# Separar alvo (y) e características (X)
X = data.drop(columns=["exam_score"])
y = data["exam_score"]

# Identificar colunas numéricas e categóricas
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Etapa 3: Transformação - Pipeline de pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Etapa 4: Mineração de dados - Pipeline com modelo
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo

model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = 1 - (mae / y_test.mean())

mae, r2, accuracy

print("📊 Avaliação do Modelo de Regressão")
print(f"🔹 Erro Médio Absoluto (MAE): {mae:.2f} pontos")
print(f"🔹 Coeficiente de Determinação (R²): {r2:.2%}")
print(f"🔹 Acurácia Aproximada: {accuracy:.2%}")

