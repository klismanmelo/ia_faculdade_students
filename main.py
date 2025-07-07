import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Carregar o dataset
file_path = "dataset.csv"
df = pd.read_csv(file_path)


# Copiar o DataFrame original para preservar os dados
data = df.copy()

# Converter a coluna de score para numérico (pode haver strings)
data['exam_score'] = pd.to_numeric(data['exam_score'], errors='coerce')

# Remover colunas irrelevantes para a predição
data = data.drop(columns=['student_id'])

# Tratar valores ausentes
data = data.dropna()

# Codificar variáveis categóricas
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separar recursos e alvo
X = data.drop(columns=['exam_score'])
y = data['exam_score']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo de regressão
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões e avaliar
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mae, r2


accuracy = 1 - (mae / y_test.mean())
print(f"Acurácia aproximada: {accuracy:.2%}")