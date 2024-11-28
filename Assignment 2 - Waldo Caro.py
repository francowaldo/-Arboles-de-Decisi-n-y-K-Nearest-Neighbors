
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 1. Cargar los datos
df = pd.read_csv('dataset.csv', sep=';')

# 2. Análisis exploratorio detallado
print("="*50)
print("ANÁLISIS EXPLORATORIO DE DATOS")
print("="*50)

# 2.1 Información básica del dataset
print("\nInformación básica del dataset:")
print("-"*30)
print(df.info())

# 2.2 Primeras filas
print("\nPrimeras 5 filas del dataset:")
print("-"*30)
print(df.head())

# 2.3 Estadísticas descriptivas detalladas
print("\nEstadísticas descriptivas:")
print("-"*30)
print(df.describe().round(2))

# 2.4 Análisis de valores únicos y distribución de especies
print("\nDistribución de especies:")
print("-"*30)
species_distribution = df['species'].value_counts()
print(species_distribution)
print("\nPorcentaje por especie:")
print((species_distribution/len(df)*100).round(2), "%")

# 2.5 Verificar valores nulos
print("\nValores nulos en el dataset:")
print("-"*30)
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print("No hay valores nulos en el dataset")
else:
    print(null_counts)

# 3. Visualizaciones básicas con matplotlib

# 3.1 Histogramas de características
plt.figure(figsize=(15, 10))
for i, feature in enumerate(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']):
    plt.subplot(2, 2, i+1)
    for species in df['species'].unique():
        plt.hist(df[df['species'] == species][feature], 
                bins=20, 
                alpha=0.5, 
                label=species)
    plt.title(f'Distribución de {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frecuencia')
    plt.legend()
plt.tight_layout()
plt.show()

# 3.2 Scatter plots de características importantes
plt.figure(figsize=(15, 5))
# Petal length vs Petal width
plt.subplot(1, 2, 1)
for species in df['species'].unique():
    mask = df['species'] == species
    plt.scatter(df[mask]['petal_length'], 
               df[mask]['petal_width'],
               label=species,
               alpha=0.6)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Petal Length vs Petal Width')
plt.legend()

# Sepal length vs Sepal width
plt.subplot(1, 2, 2)
for species in df['species'].unique():
    mask = df['species'] == species
    plt.scatter(df[mask]['sepal_length'], 
               df[mask]['sepal_width'],
               label=species,
               alpha=0.6)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Sepal Width')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Preparación de datos para el modelado
X = df.drop('species', axis=1)
y = df['species']

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print("\nPreparación de datos completada:")
print("-"*30)
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}")
print(f"\nDistribución de clases en entrenamiento:")
print(y_train.value_counts(normalize=True).round(3) * 100, "%")
print(f"\nDistribución de clases en prueba:")
print(y_test.value_counts(normalize=True).round(3) * 100, "%")

# 5. Cálculo de correlaciones
correlation_matrix = X.corr()
print("\nMatriz de correlación:")
print("-"*30)
print(correlation_matrix.round(2))

# 6. Implementación del Árbol de Decisión
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("\nImplementación del Árbol de Decisión:")
print("-"*30)

# 6.1 Crear y entrenar el modelo
dt_classifier = DecisionTreeClassifier(
    max_depth=4,          
    min_samples_split=5,  
    min_samples_leaf=2,   
    random_state=42       
)

dt_classifier.fit(X_train, y_train)

# 6.2 Realizar predicciones
y_pred = dt_classifier.predict(X_test)

# 6.3 Evaluar el modelo
print("\nMétricas de evaluación:")
print("-"*30)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# 6.4 Matriz de confusión
print("\nMatriz de confusión:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# 7. Visualización del árbol
plt.figure(figsize=(20,10))
plot_tree(dt_classifier, 
         feature_names=X.columns,
         class_names=dt_classifier.classes_,
         filled=True,
         rounded=True,
         fontsize=10)
plt.title('Árbol de Decisión para Clasificación de Iris')
plt.show()

# 8. Análisis de características importantes
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
})
importances = importances.sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
plt.bar(importances['feature'], importances['importance'])
plt.title('Importancia de las Características')
plt.xlabel('Características')
plt.ylabel('Importancia')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. Análisis detallado de predicciones
print("\nAnálisis detallado de predicciones:")
print("-"*30)
sample_indices = np.random.choice(len(X_test), 5, replace=False)
sample_X = X_test.iloc[sample_indices]
sample_y_true = y_test.iloc[sample_indices]
sample_y_pred = dt_classifier.predict(sample_X)

for i, (true, pred) in enumerate(zip(sample_y_true, sample_y_pred)):
    print(f"\nMuestra {i+1}:")
    print(f"Valores de características: {sample_X.iloc[i].to_dict()}")
    print(f"Clase verdadera: {true}")
    print(f"Clase predicha: {pred}")
    print(f"¿Predicción correcta?: {'Sí' if true == pred else 'No'}")

# 10. Validación cruzada
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(dt_classifier, X, y, cv=5)
print("\nResultados de validación cruzada:")
print("-"*30)
print(f"Accuracy promedio: {cv_scores.mean():.4f}")
print(f"Desviación estándar: {cv_scores.std():.4f}")

# 11. Implementación de K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV

print("\nImplementación de KNN:")
print("-"*30)

# 11.1 Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 11.2 Encontrar el mejor valor de K
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# Visualizar los resultados de diferentes valores de K
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, marker='o')
plt.xlabel('Valor de K')
plt.ylabel('Accuracy promedio')
plt.title('Accuracy vs K Value')
plt.grid(True)
plt.show()

# 11.3 Seleccionar el mejor K
best_k = k_range[np.argmax(k_scores)]
print(f"\nMejor valor de K encontrado: {best_k}")

# 11.4 Entrenar el modelo con el mejor K
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)

# 11.5 Realizar predicciones
y_pred_knn = best_knn.predict(X_test_scaled)

# 12. Evaluación del modelo KNN
print("\nEvaluación del modelo KNN:")
print("-"*30)

# 12.1 Métricas básicas
print("\nAccuracy:", round(accuracy_score(y_test, y_pred_knn), 4))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_knn))

# 12.2 Matriz de confusión
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
print("\nMatriz de confusión:")
print(conf_matrix_knn)

# 13. Visualización de las predicciones
plt.figure(figsize=(12, 5))

# 13.1 Visualizar predicciones en espacio de características principales
# Convertir etiquetas a números para la visualización
le = LabelEncoder()
y_pred_encoded = le.fit_transform(y_pred_knn)

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_test_scaled[:, 2], X_test_scaled[:, 3], 
                     c=y_pred_encoded, 
                     cmap='viridis')
plt.xlabel('Petal Length (scaled)')
plt.ylabel('Petal Width (scaled)')
plt.title('Predicciones KNN')
plt.colorbar(scatter)

# 13.2 Visualizar regiones de decisión
plt.subplot(1, 2, 2)
x_min, x_max = X_test_scaled[:, 2].min() - 1, X_test_scaled[:, 2].max() + 1
y_min, y_max = X_test_scaled[:, 3].min() - 1, X_test_scaled[:, 3].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = best_knn.predict(np.c_[np.zeros_like(xx.ravel()), 
                          np.zeros_like(xx.ravel()),
                          xx.ravel(), 
                          yy.ravel()])
Z = le.transform(Z)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.scatter(X_test_scaled[:, 2], X_test_scaled[:, 3], 
           c=y_pred_encoded, 
           cmap='viridis')
plt.xlabel('Petal Length (scaled)')
plt.ylabel('Petal Width (scaled)')
plt.title('Regiones de Decisión KNN')
plt.tight_layout()
plt.show()

# 14. Análisis detallado de predicciones KNN
print("\nAnálisis detallado de predicciones KNN:")
print("-"*30)
sample_indices = np.random.choice(len(X_test), 5, replace=False)
sample_X = X_test_scaled[sample_indices]
sample_y_true = y_test.iloc[sample_indices]
sample_y_pred = best_knn.predict(sample_X)

for i, (true, pred) in enumerate(zip(sample_y_true, sample_y_pred)):
    print(f"\nMuestra {i+1}:")
    neighbors = best_knn.kneighbors([sample_X[i]], return_distance=True)
    print(f"Clase verdadera: {true}")
    print(f"Clase predicha: {pred}")
    print(f"¿Predicción correcta?: {'Sí' if true == pred else 'No'}")
    print(f"Distancia a los {best_k} vecinos más cercanos: {neighbors[0][0].round(3)}")

# 15. Validación cruzada final
cv_scores_knn = cross_val_score(best_knn, X_train_scaled, y_train, cv=5)
print("\nResultados de validación cruzada KNN:")
print("-"*30)
print(f"Accuracy promedio: {cv_scores_knn.mean():.4f}")
print(f"Desviación estándar: {cv_scores_knn.std():.4f}")

# 16. Comparación de modelos (Árbol de Decisión vs KNN)
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from itertools import cycle

print("\nComparación de modelos:")
print("="*50)

# 16.1 Tabla comparativa de métricas
print("\nMétricas comparativas:")
print("-"*30)

# Calcular métricas para ambos modelos
models_comparison = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precisión promedio', 'Recall promedio', 'F1-Score promedio'],
    'Árbol de Decisión': [
        accuracy_score(y_test, y_pred),
        classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision'],
        classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall'],
        classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    ],
    'KNN': [
        accuracy_score(y_test, y_pred_knn),
        classification_report(y_test, y_pred_knn, output_dict=True)['weighted avg']['precision'],
        classification_report(y_test, y_pred_knn, output_dict=True)['weighted avg']['recall'],
        classification_report(y_test, y_pred_knn, output_dict=True)['weighted avg']['f1-score']
    ]
})

print(models_comparison.round(4))

# 16.2 Visualización comparativa de matrices de confusión
plt.figure(figsize=(15, 5))

# Matriz de confusión para el Árbol de Decisión
plt.subplot(1, 2, 1)
conf_matrix_dt = confusion_matrix(y_test, y_pred)
plt.imshow(conf_matrix_dt, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Árbol de Decisión')
plt.colorbar()

# Añadir etiquetas numéricas
thresh = conf_matrix_dt.max() / 2.
for i, j in np.ndindex(conf_matrix_dt.shape):
    plt.text(j, i, format(conf_matrix_dt[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix_dt[i, j] > thresh else "black")

# Matriz de confusión para KNN
plt.subplot(1, 2, 2)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
plt.imshow(conf_matrix_knn, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - KNN')
plt.colorbar()

# Añadir etiquetas numéricas
thresh = conf_matrix_knn.max() / 2.
for i, j in np.ndindex(conf_matrix_knn.shape):
    plt.text(j, i, format(conf_matrix_knn[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix_knn[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# 16.3 Análisis detallado de errores
print("\nAnálisis de errores:")
print("-"*30)

# Crear máscaras para predicciones diferentes
mask_diferentes = y_pred != y_pred_knn
indices_diferentes = np.where(mask_diferentes)[0]

print(f"\nNúmero de casos donde los modelos difieren: {len(indices_diferentes)}")

if len(indices_diferentes) > 0:
    print("\nEjemplos de predicciones diferentes:")
    for i in indices_diferentes[:3]:  # Tomamos los primeros 3 casos
        print(f"\nMuestra {i}:")
        # Convertir los datos a un diccionario más legible
        muestra = {
            'sepal_length': X_test.iloc[i,0],
            'sepal_width': X_test.iloc[i,1],
            'petal_length': X_test.iloc[i,2],
            'petal_width': X_test.iloc[i,3]
        }
        print("Valores de las características:")
        for caracteristica, valor in muestra.items():
            print(f"- {caracteristica}: {valor:.2f}")
        print(f"Clase verdadera: {y_test.iloc[i]}")
        print(f"Predicción Árbol: {y_pred[i]}")
        print(f"Predicción KNN: {y_pred_knn[i]}")

# 17. Conclusiones
print("\nConclusiones:")
print("="*50)

# 17.1 Resumen de rendimiento
print("\nResumen de rendimiento:")
print("-"*30)
dt_accuracy = accuracy_score(y_test, y_pred)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

print(f"Accuracy Árbol de Decisión: {dt_accuracy:.4f}")
print(f"Accuracy KNN: {knn_accuracy:.4f}")

if dt_accuracy > knn_accuracy:
    mejor_modelo = "Árbol de Decisión"
    diferencia = (dt_accuracy - knn_accuracy) * 100
else:
    mejor_modelo = "KNN"
    diferencia = (knn_accuracy - dt_accuracy) * 100

# 17.2 Ventajas y desventajas observadas
print("\nAnálisis cualitativo de los modelos:")
print("\nÁrbol de Decisión:")
print("-"*30)
print("Ventajas:")
print("- Fácil interpretación visual del proceso de decisión")
print("- Identifica automáticamente las características más importantes")
print("- No requiere escalado de datos")
print("\nDesventajas:")
print("- Puede tender al sobreajuste si no se controla la profundidad")
print("- Sensible a pequeños cambios en los datos")
print("- Fronteras de decisión limitadas a divisiones ortogonales")

print("\nKNN:")
print("-"*30)
print("Ventajas:")
print("- No hace suposiciones sobre la distribución de los datos")
print("- Puede capturar patrones complejos")
print("- Se adapta bien a nuevos datos")
print("\nDesventajas:")
print("- Requiere escalado de características")
print("- Computacionalmente más costoso en la fase de predicción")
print("- Sensible al ruido en los datos")

# 17.3 Recomendación final
print("\nRecomendación final:")
print("-"*30)
print(f"El modelo con mejor rendimiento es: {mejor_modelo}")
print(f"Diferencia en accuracy: {diferencia:.2f}%")

if mejor_modelo == "Árbol de Decisión":
    print("\nSe recomienda usar el Árbol de Decisión por:")
    print("- Mayor interpretabilidad")
    print("- Mejor rendimiento en los datos de prueba")
    print("- Menor costo computacional en predicciones")
else:
    print("\nSe recomienda usar KNN por:")
    print("- Mejor capacidad de generalización")
    print("- Mejor rendimiento en los datos de prueba")
    print("- Mayor flexibilidad en las fronteras de decisión")