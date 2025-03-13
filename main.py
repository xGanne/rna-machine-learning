import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

data = pd.read_csv('data_cancer2.csv')
data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model.__class__.__name__} - Accuracy:", accuracy_score(y_test, y_pred))
    print(f"{model.__class__.__name__} - Classification Report:\n", classification_report(y_test, y_pred))
    print(f"{model.__class__.__name__} - Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Rede Neural Artificial
mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=500, random_state=42)
train_and_evaluate(mlp, X_train, X_test, y_train, y_test)

# Random Forest
rf = RandomForestClassifier(random_state=42)
train_and_evaluate(rf, X_train, X_test, y_train, y_test)

# Support Vector Machine
svc = SVC(random_state=42)
train_and_evaluate(svc, X_train, X_test, y_train, y_test)

# Regressão Logística
lr = LogisticRegression(random_state=42, max_iter=500)
train_and_evaluate(lr, X_train, X_test, y_train, y_test)

# K-Nearest Neighbors
knn = KNeighborsClassifier()
train_and_evaluate(knn, X_train, X_test, y_train, y_test)

# Árvore de Decisão
dt = DecisionTreeClassifier(random_state=42)
train_and_evaluate(dt, X_train, X_test, y_train, y_test)