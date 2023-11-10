# Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree


# Vamos a generar un conjunto de datos de ejemplo
num_muestras = 100
Gasto_Publicidad_ATL = np.random.uniform(10000, 50000, num_muestras)
num_ATL_1 = np.random.randint(5, 20, num_muestras)
num_ATL_2 = np.random.randint(0, 3, num_muestras)
brand_awareness = 50 + 2 * Gasto_Publicidad_ATL + 3 * num_ATL_2 + 10 * num_ATL_2 + np.random.normal(0, 5000, num_muestras) 
#Para el brand awareness se sugiere hacer regresiones para llegar a una medición en el paso 1

#  DataFrame con los datos de protitipo (ejemplo con ATL).
datos = pd.DataFrame({
    'Gasto_Publicidad_ATL': Gasto_Publicidad_ATL,
    'Metricas_ATL_1': num_ATL_1,
    'num_ATL_2': num_ATL_2,
    'Brand_Awareness': brand_awareness
})

# Con esto dividimos el conjunto de datos en entrenamiento y prueba
X = datos[['Gasto_Publicidad_ATL', 'num_ATL_2', 'num_ATL_2']]
y = datos['Brand_Awareness']
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Acá se crea y entrena el modelo de árbol de regresión
modelo_arbol_regresion = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)
modelo_arbol_regresion.fit(X_entrenamiento, y_entrenamiento)

# Posterior realizamos predicciones en el conjunto de prueba
predicciones = modelo_arbol_regresion.predict(X_prueba)

# Evaluamos el rendimiento del modelo
mse = mean_squared_error(y_prueba, predicciones)
r2 = r2_score(y_prueba, predicciones)

# Vamos a evaluar métricas estadísticas para ver la presición
print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Coeficiente de Determinación (R^2): {r2}")

# Vamos a visualizar el modelo y sus predicciones
plt.figure(figsize=(10, 6))
plt.scatter(X_prueba['Gasto_Publicidad_ATL'], y_prueba, color='black', label='Datos reales')
plt.scatter(X_prueba['Gasto_Publicidad_ATL'], predicciones, color='blue', label='Predicciones')
plt.title('Árbol de Regresión - Brand Awareness')
plt.xlabel('Gasto en Publicidad en Medio ATL')
plt.ylabel('Brand Awareness')
plt.legend()
plt.show()

# Por último visualizamos el árbol de regresión
plt.figure(figsize=(15, 10))
plot_tree(modelo_arbol_regresion, feature_names=X.columns, filled=True, rounded=True, fontsize=10, precision=2)
plt.title('Árbol de Regresión')
plt.show()
