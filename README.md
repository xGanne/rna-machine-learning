# Classificação de Tumores de Mama com Algoritmos de Machine Learning

## Descrição do Projeto

Este projeto tem como objetivo desenvolver e comparar diferentes algoritmos de classificação para determinar se um tumor de mama é maligno ou benigno. Utilizamos o dataset `data_cancer2.csv`, que contém características computadas de imagens digitalizadas de células de câncer de mama.

## Algoritmos Utilizados

Os seguintes algoritmos de classificação foram implementados e avaliados:

1. **Rede Neural Artificial (MLPClassifier)**
2. **Random Forest (RandomForestClassifier)**
3. **Support Vector Machine (SVC)**
4. **Regressão Logística (LogisticRegression)**
5. **K-Nearest Neighbors (KNeighborsClassifier)**
6. **Árvore de Decisão (DecisionTreeClassifier)**

## Pré-processamento dos Dados

1. **Carregamento do Dataset**: O dataset foi carregado a partir do arquivo `data_cancer2.csv`.
2. **Remoção de Colunas Irrelevantes**: As colunas `id` e `Unnamed: 32` foram removidas.
3. **Conversão de Rótulos**: A coluna `diagnosis` foi convertida para valores binários (`M` para 1 e `B` para 0).
4. **Tratamento de Valores Ausentes**: Valores ausentes foram substituídos pela média das colunas.
5. **Divisão dos Dados**: Os dados foram divididos em conjuntos de treino (80%) e teste (20%).
6. **Padronização dos Dados**: As features foram padronizadas para terem média 0 e desvio padrão 1.

## Resultados dos Modelos

### Rede Neural Artificial (MLPClassifier)
- **Acurácia**: 96.49%
- **Relatório de Classificação**:
### Rede Neural Artificial (MLPClassifier)
- **Acurácia**: 96.49%
- **Relatório de Classificação**:
    ```
                             precision    recall  f1-score   support

                     0       0.97      0.97      0.97        71
                     1       0.96      0.96      0.96        43

        accuracy                           0.96       114
     macro avg       0.96      0.96      0.96       114
weighted avg       0.96      0.96      0.96       114
    ```
- **Matriz de Confusão**:
    ```
    [[69  2]
     [ 2 41]]
    ```

### Observações
O algoritmo de Redes Neurais Artificiais (MLPClassifier) apresentou um desempenho excelente, com uma acurácia de 96.49%. A matriz de confusão mostra que o modelo cometeu poucos erros, classificando corretamente a maioria dos tumores malignos e benignos.

### Comparação de Desempenho
Para determinar se o algoritmo de Redes Neurais Artificiais superou os demais algoritmos de classificação, comparamos as acurácias e os relatórios de classificação de todos os modelos implementados. Abaixo estão os resultados dos outros algoritmos:

- **Random Forest**: Acurácia de 95.61%
- **Support Vector Machine**: Acurácia de 96.49%
- **Regressão Logística**: Acurácia de 95.61%
- **K-Nearest Neighbors**: Acurácia de 94.74%
- **Árvore de Decisão**: Acurácia de 92.98%

### Conclusão
O algoritmo de Redes Neurais Artificiais conseguiu igualar o desempenho do algoritmo de Support Vector Machine, ambos atingindo a melhor acurácia de 96.49%. Portanto, o MLPClassifier foi um dos melhores algoritmos de classificação utilizados neste projeto, demonstrando alta precisão e capacidade de generalização.