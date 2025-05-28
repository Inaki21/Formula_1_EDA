# Cargamos los datos iniciales y las bibliotecas necesarias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
#%matplotlib inline
import os
os.makedirs("../img/Graficas_generadas/", exist_ok=True)
os.makedirs("..Datos/Datos_csv_creados/", exist_ok=True)

drivers_df = pd.read_csv('../Datos/Datos_csv/drivers.csv')
races_df = pd.read_csv('../Datos/Datos_csv/races.csv')
qualifying_df = pd.read_csv('../Datos/Datos_csv/qualifying.csv')
races_df = pd.read_csv('../Datos/Datos_csv/races.csv')
results_df = pd.read_csv('../Datos/Datos_csv/results.csv')
constructors_df = pd.read_csv('../Datos/Datos_csv/constructors.csv')
circuits_df = pd.read_csv('../Datos/Datos_csv/circuits.csv')
status_df = pd.read_csv('../Datos/Datos_csv/status.csv')
lap_times_df = pd.read_csv('../Datos/Datos_csv/lap_times.csv')
pit_stops_df = pd.read_csv('../Datos/Datos_csv/pit_stops.csv')
constructor_standings_df = pd.read_csv('../Datos/Datos_csv/constructor_standings.csv')


drivers = drivers_df 
races = races_df
qualifying = qualifying_df 
races = races_df 
results = results_df 
constructors = constructors_df 
circuits = circuits_df 
status = status_df 
lap_times = lap_times_df 
pit_stop = pit_stops_df 
constructor_standings = constructor_standings_df


# Vamos a reconstruir el dataframe que necesitamos incluyendo toda la información relevante
# explicando paso a paso

# Paso 1: drivers - información del piloto
drivers_df = drivers[['driverId', 'driverRef', 'forename', 'surname', 'nationality']]

# Paso 2: results - información final de la carrera (posición final, grid, puntos, constructorId)
results_df = results[['driverId', 'raceId', 'constructorId', 'grid', 'position', 'points', 'statusId']]

# Paso 3: qualifying - posición inicial de clasificación
qualifying_df = qualifying[['driverId', 'raceId', 'position']].rename(columns={'position': 'qualifying_position'})

# Paso 4: races - información de la carrera (año, ronda, circuito, nombre, fecha)
races_df = races[['raceId', 'year', 'round', 'circuitId', 'name', 'date']]

# Paso 5: circuits - ubicación y país del circuito
circuits_df = circuits[['circuitId', 'name', 'location', 'country']].rename(columns={'name': 'circuit_name'})

# Paso 6: constructors - nombre y nacionalidad del constructor
constructors_df = constructors[['constructorId', 'name', 'nationality']].rename(columns={'name': 'constructor_name', 'nationality': 'constructor_nationality'})

# Paso 7: Años en los que cada constructor participó
years_constructor = constructor_standings[['constructorId', 'raceId']].merge(races[['raceId', 'year']], on='raceId', how='left')
years_by_constructor = years_constructor.groupby('constructorId')['year'].unique().reset_index().rename(columns={'year': 'years_competed'})

# Paso 8: status - descripción del resultado final
status_df = status[['statusId', 'status']]

# Ahora realizamos los merges progresivos
# Merge de drivers con results (posición final, puntos, etc.)
df_merge = drivers_df.merge(results_df, on='driverId', how='left')

# Merge con qualifying para posición de clasificación (inicial)
df_merge = df_merge.merge(qualifying_df, on=['driverId', 'raceId'], how='left')

# Merge con races para datos de la carrera
df_merge = df_merge.merge(races_df, on='raceId', how='left')

# Merge con circuits para país y localización
df_merge = df_merge.merge(circuits_df, on='circuitId', how='left')

# Merge con constructors para info del constructor
df_merge = df_merge.merge(constructors_df, on='constructorId', how='left')

# Merge con los años en que compitieron los constructores
df_merge = df_merge.merge(years_by_constructor, on='constructorId', how='left')

# Merge con status para descripción del resultado final
df_merge = df_merge.merge(status_df, on='statusId', how='left')
df_pilotos_completo = df_merge ###################3

# Filtramos el dataframe para incluir solo a los pilotos de nacionalidad española
df_espanoles = df_merge[df_merge['nationality'] == 'Spanish']

# Guardamos este dataframe como un CSV
df_espanoles.to_csv('../Datos/Datos_csv_creados/_pilotos_espanoles_f1.csv', index=False)

# Contamos el número de carreras por piloto
carreras_por_piloto = df_espanoles.groupby('driverId').size().reset_index(name='num_carreras')

# Filtramos los pilotos que hayan corrido más de 5 carreras
pilotos_mas_5_carreras = carreras_por_piloto[carreras_por_piloto['num_carreras'] > 5]['driverId']

# Filtramos el dataframe para incluir solo esos pilotos
df_espanoles_mas_5_carreras = df_espanoles[df_espanoles['driverId'].isin(pilotos_mas_5_carreras)]

df_espanoles_mas_5_carreras

# Contamos el número de carreras por piloto
num_carreras = df_espanoles_mas_5_carreras.groupby(['driverId', 'forename', 'surname']).size().reset_index(name='num_carreras')

# Puntos totales por piloto
total_puntos = df_espanoles_mas_5_carreras.groupby(['driverId', 'forename', 'surname'])['points'].sum().reset_index(name='total_puntos')

# Número de victorias (posición final = 1)
victorias = df_espanoles_mas_5_carreras[df_espanoles_mas_5_carreras['position'] == 1].groupby(['driverId', 'forename', 'surname']).size().reset_index(name='num_victorias')

# Años en los que compitieron
years_participacion = df_espanoles_mas_5_carreras.groupby(['driverId', 'forename', 'surname'])['year'].unique().reset_index()
years_participacion['num_years'] = years_participacion['year'].apply(lambda x: len(x))

# Constructores únicos por piloto
constructores_por_piloto = df_espanoles_mas_5_carreras.groupby(['driverId', 'forename', 'surname'])['constructor_name'].unique().reset_index()
constructores_por_piloto['constructors'] = constructores_por_piloto['constructor_name'].apply(lambda x: ', '.join(x))

# Combinamos toda la información
df_stats_espanoles = num_carreras.merge(total_puntos, on=['driverId', 'forename', 'surname'], how='left')
df_stats_espanoles = df_stats_espanoles.merge(victorias, on=['driverId', 'forename', 'surname'], how='left')
df_stats_espanoles = df_stats_espanoles.merge(years_participacion[['driverId', 'year', 'num_years']], on='driverId', how='left')
df_stats_espanoles = df_stats_espanoles.merge(constructores_por_piloto[['driverId', 'constructors']], on='driverId', how='left')

# Rellenamos NaN de victorias con 0
df_stats_espanoles['num_victorias'] = df_stats_espanoles['num_victorias'].fillna(0).astype(int)
# Renombramos columna para mayor claridad
df_stats_espanoles = df_stats_espanoles.rename(columns={'year': 'years_participated'})
df_stats_espanoles
df_stats_espanoles = df_stats_espanoles.sort_values(by='total_puntos', ascending=False)
df_stats_espanoles.to_csv('../Datos/Datos_csv_creados/_df_stats_espanoles.csv', index=False)

##########################GRAFICA 1


# Preparamos los datos y el gráfico de barras con leyenda de pilotos y puntos totales
plt.figure(figsize=(10, 6))
bars = plt.bar(df_stats_espanoles['surname'], df_stats_espanoles['total_puntos'], color='skyblue')
plt.xlabel('Piloto')
plt.ylabel('Puntos Totales')
plt.title('Puntos totales por piloto español en F1')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Añadimos la leyenda como anotaciones encima de cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f'{int(yval)}', ha='center', va='bottom', fontsize=9)
# Preparamos la leyenda personalizada con piloto y puntos
legend_labels = [f"{name}: {int(points)} puntos" for name, points in zip(df_stats_espanoles['surname'], df_stats_espanoles['total_puntos'])]

# Añadimos la leyenda fuera del gráfico
plt.legend(bars, legend_labels, title='Pilotos y Puntos Totales', loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/1_grafica.png")
# plt.show()
##########################GRAFICA 2
# Creamos el gráfico de barras horizontales con leyenda externa
plt.figure(figsize=(10, 6))
bars = plt.barh(df_stats_espanoles['surname'], df_stats_espanoles['num_carreras'], color='lightgreen')
plt.xlabel('Número de carreras')
plt.ylabel('Piloto')
plt.title('Número de carreras disputadas por piloto español en F1')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Añadimos los años de participación a la derecha de cada barra
for i, (bar, years) in enumerate(zip(bars, df_stats_espanoles['years_participated'])):
    plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f"Años: {len(years)}", va='center', fontsize=9)

# Creamos la leyenda personalizada con piloto y años de participación
legend_labels = [f"{name}: {len(years)} años" for name, years in zip(df_stats_espanoles['surname'], df_stats_espanoles['years_participated'])]

# Añadimos la leyenda fuera del gráfico
plt.legend(bars, legend_labels, title='Pilotos y años de participación', loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig("../img/Graficas_generadas/2_grafica.png")

# plt.show()


##########################GRAFICA 3

df_espanoles['position'] = pd.to_numeric(df_espanoles['position'], errors='coerce')

# Filtramos solo las carreras donde la posición final es 1 
df_victorias = df_espanoles[df_espanoles['position'] == 1]

#Contamos el número de victorias por piloto (usamos el apellido/surname)
victorias_por_piloto = df_victorias.groupby('surname').size().reset_index(name='num_victorias')

# Creamos la gráfica de barras
plt.figure(figsize=(8, 5))
bars = plt.bar(victorias_por_piloto['surname'], victorias_por_piloto['num_victorias'], color='salmon')
plt.xlabel('Piloto')
plt.ylabel('Número de victorias')
plt.title('Número de victorias por piloto español en F1 (al menos una victoria)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Leyenda externa con número de victorias
legend_labels = [f"{name}: {victories} victorias" for name, victories in zip(victorias_por_piloto['surname'], victorias_por_piloto['num_victorias'])]
plt.legend(bars, legend_labels, title='Pilotos y victorias', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

plt.tight_layout()
plt.savefig("../img/Graficas_generadas/3_grafica.png")
plt.show()

##########################GRAFICA 4
# Para cada piloto, vamos a graficar los años en los que participó

plt.figure(figsize=(12, 6))

# Para cada piloto en df_stats_espanoles
for i, row in df_stats_espanoles.iterrows():
    surname = row['surname']
    years = sorted(row['years_participated'])
    plt.plot(years, [surname] * len(years), marker='o', linestyle='-', label=surname)
    #plt.text(years[-1] + 0.3, surname, surname, va='center', fontsize=9, color='black')
plt.xlabel('Año')
plt.ylabel('Piloto')
plt.title('Años de participación en F1 por piloto español')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Añadimos la leyenda fuera del gráfico
plt.legend(title='Pilotos', loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig("../img/Graficas_generadas/4_grafica.png")

# plt.show()
##########################GRAFICA 5

# Primero, creamos un dataframe de ocurrencias: 1 si el piloto ha corrido con el constructor, 0 si no
pilotos = df_stats_espanoles['surname']
constructores = sorted(set(sum([row.split(', ') for row in df_stats_espanoles['constructors']], [])))

# Creamos un dataframe vacío con 0
heatmap_data = pd.DataFrame(0, index=pilotos, columns=constructores)

# Rellenamos la matriz con 1 donde el piloto ha participado con ese constructor
for i, row in df_stats_espanoles.iterrows():
    surname = row['surname']
    pilotos_constructores = row['constructors'].split(', ')
    for constructor in pilotos_constructores:
        heatmap_data.loc[surname, constructor] = 1

# Creamos el heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', cbar=False, linewidths=0.5, linecolor='lightgray')
plt.title('Constructores con los que ha corrido cada piloto español')
plt.xlabel('Constructor')
plt.ylabel('Piloto')
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/5_grafica.png")

# plt.show()
#############################################33 GRAFICA6




#######################

# Calculamos la eficiencia de puntos por carrera
df_stats_espanoles['puntos_por_carrera'] = df_stats_espanoles['total_puntos'] / df_stats_espanoles['num_carreras']

# Creamos el scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(df_stats_espanoles['num_carreras'], df_stats_espanoles['total_puntos'], color='purple', alpha=0.7)

# Etiquetamos a cada piloto en el scatterplot
for i, row in df_stats_espanoles.iterrows():
    plt.text(row['num_carreras'] + 2, row['total_puntos'], row['surname'], fontsize=8)

# Creamos leyenda externa con piloto y puntos totales
legend_labels = [f"{name}: {int(points)} puntos" for name, points in zip(df_stats_espanoles['surname'], df_stats_espanoles['total_puntos'])]
plt.legend(legend_labels, title='Pilotos y puntos totales', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)


plt.xlabel('Número de carreras')
plt.ylabel('Puntos totales')
plt.title('Puntos totales vs. número de carreras (eficiencia)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/7_grafica.png")

# plt.show()
##########################GRAFICA 7
##################################################
# Usamos la columna 'nationality' del CSV de pilotos para quedarnos solo con los españoles
spanish_drivers = drivers[drivers['nationality'] == 'Spanish']

# Mostramos los pilotos españoles encontrados para confirmar los datos
print("Pilotos españoles encontrados:")
print(spanish_drivers[['driverId', 'driverRef', 'surname']])

# Paso 3: Unimos los datos para obtener la información completa de los resultados de cada piloto español
# 3.1: Unimos 'results' con 'drivers' usando 'driverId' para tener datos básicos del piloto en cada resultado
spanish_results = pd.merge(results, spanish_drivers[['driverId', 'surname']], on='driverId')

# 3.2: Unimos con 'races' usando 'raceId' para obtener año y nombre de la carrera
spanish_results = pd.merge(spanish_results, races[['raceId', 'year', 'name']], on='raceId')

# 3.3: Unimos con 'constructors' usando 'constructorId' para saber el equipo con el que compitió
spanish_results = pd.merge(spanish_results, constructors[['constructorId', 'name']], on='constructorId', suffixes=('', '_constructor'))

# Paso 4: Seleccionamos las columnas clave para el análisis final
# Incluimos 'driverId' para identificar de manera única al piloto
spanish_results_df = spanish_results[['driverId', 'surname', 'year', 'name', 'grid', 'positionOrder', 'points', 'name_constructor']]

# Renombramos las columnas para mayor claridad
spanish_results_df.columns = ['Piloto_ID', 'Piloto', 'Año', 'Carrera', 'Posición_Salida', 'Posición_Final', 'Puntos', 'Constructor']

# Paso 5: Guardamos esta tabla en un CSV para usarla más adelante en la presentación
#spanish_results_df.to_csv("Datos_csv_creados/spanish_drivers_results.csv", index=False)

# Mostramos las primeras filas para verificar el resultado
#print("\nDatos finales de los pilotos españoles:")
#print(spanish_results_df.head())

# Paso 6: Calculamos el número de carreras en las que participó cada piloto español
# Agrupamos por 'Piloto_ID' y 'Piloto' para contar las filas (carreras)
carreras_por_piloto = spanish_results_df.groupby(['Piloto_ID', 'Piloto']).size().reset_index(name='Numero_Carreras')

# Mostramos la tabla para confirmar el número de carreras por piloto
#print("\nNúmero de carreras por piloto español:")
#print(carreras_por_piloto)

# Paso 7: Calculamos los puntos totales de cada piloto español
# Agrupamos por 'Piloto_ID' y 'Piloto' y sumamos la columna 'Puntos'
puntos_por_piloto = spanish_results_df.groupby(['Piloto_ID', 'Piloto'])['Puntos'].sum().reset_index(name='Puntos_Totales')

# Mostramos la tabla de puntos totales por piloto para comprobar los datos
#print("\nPuntos totales por piloto español:")
#print(puntos_por_piloto)

# Paso 8: Unimos las tablas de número de carreras y puntos totales para ver la relación
# Usamos 'Piloto_ID' y 'Piloto' como llaves
resumen_pilotos = pd.merge(carreras_por_piloto, puntos_por_piloto, on=['Piloto_ID', 'Piloto'])

resumen_pilotos = pd.read_csv("../Datos/Datos_csv_creados/spanish_drivers_carreras_puntos.csv")

##########################GRAFICA 9
# Creamos un DataFrame auxiliar que combina el nombre del piloto y sus puntos totales para la leyenda
resumen_pilotos['Piloto_Leyenda'] = resumen_pilotos.apply(
    lambda row: f"{row['Piloto']} ({int(row['Puntos_Totales'])} pts)", axis=1
)

# Gráfico scatter plot
plt.figure(figsize=(10, 6))

# Dibujamos el scatter plot usando 'Piloto_Leyenda' como leyenda
# De esta manera, los puntos en el gráfico tienen la leyenda de 'piloto + puntos totales'
sns.scatterplot(data=resumen_pilotos,
                x='Numero_Carreras',
                y='Puntos_Totales',
                hue='Piloto_Leyenda',
                s=100,
                palette='deep')

plt.title('Relación entre número de carreras y puntos totales (pilotos españoles)')
plt.xlabel('Número de carreras')
plt.ylabel('Puntos totales')
plt.grid(True)
plt.tight_layout()

# Guardamos la gráfica en la carpeta img
plt.savefig("../img/Graficas_generadas/8_grafica.png")

# plt.show()
###################################################################################

# Aseguramos que las columnas están en formato numérico
df_espanoles_mas_5_carreras['grid'] = pd.to_numeric(df_espanoles_mas_5_carreras['grid'], errors='coerce')
df_espanoles_mas_5_carreras['position'] = pd.to_numeric(df_espanoles_mas_5_carreras['position'], errors='coerce')

# Creamos la columna diferencia_posiciones
df_espanoles_mas_5_carreras['diferencia_posiciones'] = df_espanoles_mas_5_carreras['grid'] - df_espanoles_mas_5_carreras['position']

# Ahora filtramos solo a Alonso
df_alonso = df_espanoles_mas_5_carreras[df_espanoles_mas_5_carreras['driverId'] == 4]

# Nos quedamos con las columnas relevantes
df_alonso = df_alonso[['driverId', 'surname', 'raceId', 'name', 'year', 'grid', 'position', 'diferencia_posiciones']]

# Verificamos la posición máxima de salida (grid) en el dataframe de Alonso
max_grid = df_alonso['grid'].max()

# Primero, generamos la gráfica manteniendo los NaN pero reemplazándolos por la posición máxima de salida
df_alonso_reemplazo = df_alonso.copy()
df_alonso_reemplazo['position_reemplazo'] = df_alonso_reemplazo['position'].fillna(max_grid)
df_alonso_reemplazo['diferencia_posiciones_reemplazo'] = df_alonso_reemplazo['grid'] - df_alonso_reemplazo['position_reemplazo']
# Segundo, generamos la gráfica solo con las carreras que terminó (sin NaN)
df_alonso_terminadas = df_alonso.dropna(subset=['position']).copy()
df_alonso_terminadas['diferencia_posiciones'] = df_alonso_terminadas['grid'] - df_alonso_terminadas['position']

# Boxplot - carreras con NaN reemplazados
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_alonso_reemplazo, x='year', y='diferencia_posiciones_reemplazo', color='skyblue')
plt.title('Boxplot - Posiciones ganadas/perdidas con NaN reemplazados')
plt.ylabel('Posiciones ganadas/perdidas')
plt.xlabel('Año')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/9_grafica.png")

# plt.show()

##########################GRAFICA 10
# Boxplot - solo carreras terminadas
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_alonso_terminadas, x='year', y='diferencia_posiciones', color='lightgreen')
plt.title('Boxplot - Posiciones ganadas/perdidas solo carreras terminadas')
plt.ylabel('Posiciones ganadas/perdidas')
plt.xlabel('Año')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/10_grafica.png")

# plt.show()
##########################GRAFICA 11
# Primero, creamos el histograma con las diferencias de posiciones de Alonso (solo carreras terminadas)
plt.figure(figsize=(10, 6))
plt.hist(df_alonso_terminadas['diferencia_posiciones'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Cambio de posición (final - salida)')
plt.ylabel('Número de carreras')
plt.title('Distribución de mejora/empeoramiento en carrera - Alonso (solo carreras terminadas)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/11_grafica.png")

# plt.show()

# Calculamos media, mediana y moda
media_alonso = df_alonso_terminadas['diferencia_posiciones'].mean()
mediana_alonso = df_alonso_terminadas['diferencia_posiciones'].median()
moda_alonso = df_alonso_terminadas['diferencia_posiciones'].mode()[0]  # La moda puede tener múltiples valores, tomamos el primero
##########################GRAFICA 12
#Generamos el histograma de Alonso con NaN reemplazados por posición máxima de salida
plt.figure(figsize=(10, 6))
plt.hist(df_alonso_reemplazo['diferencia_posiciones_reemplazo'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Cambio de posición (final - salida)')
plt.ylabel('Número de carreras')
plt.title('Distribución de mejora/empeoramiento en carrera - Alonso (con NaN reemplazados)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/12_grafica.png")

# plt.show()
##########################GRAFICA 13
# Calculamos media, mediana y moda para todas las carreras (con NaN reemplazados)
media_alonso_all = df_alonso_reemplazo['diferencia_posiciones_reemplazo'].mean()
mediana_alonso_all = df_alonso_reemplazo['diferencia_posiciones_reemplazo'].median()
moda_alonso_all = df_alonso_reemplazo['diferencia_posiciones_reemplazo'].mode()[0]

# Calculamos la media de diferencia de posiciones por año (solo carreras terminadas)
media_anual_alonso = df_alonso_terminadas.groupby('year')['diferencia_posiciones'].mean().reset_index()

# Gráfico de línea de la evolución de la media por año
plt.figure(figsize=(10, 6))
plt.plot(media_anual_alonso['year'], media_anual_alonso['diferencia_posiciones'], marker='o', color='orange', label='Media anual')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Año')
plt.ylabel('Promedio de posiciones ganadas/perdidas')
plt.title('Evolución de la media de posiciones ganadas/perdidas por año - Alonso (carreras terminadas)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/13_grafica.png")

# plt.show()
##########################GRAFICA 14
# Filtramos las carreras de Carlos Sainz (driverId de Sainz, que normalmente es 832, pero lo verificamos primero)
driver_ids_sainz = df_espanoles_mas_5_carreras[df_espanoles_mas_5_carreras['surname'] == 'Sainz']['driverId'].unique()
driver_id_sainz = driver_ids_sainz[0]  # Tomamos el primero

# Filtramos sus carreras y solo las terminadas (position no nulo)
df_sainz = df_espanoles_mas_5_carreras[df_espanoles_mas_5_carreras['driverId'] == driver_id_sainz].copy()
df_sainz_terminadas = df_sainz.dropna(subset=['position']).copy()
df_sainz_terminadas['diferencia_posiciones'] = df_sainz_terminadas['grid'] - df_sainz_terminadas['position']

# Calculamos la media por año para Sainz
media_anual_sainz = df_sainz_terminadas.groupby('year')['diferencia_posiciones'].mean().reset_index()

# Gráfica comparativa Alonso vs Sainz
plt.figure(figsize=(10, 6))
plt.plot(media_anual_alonso['year'], media_anual_alonso['diferencia_posiciones'], marker='o', color='orange', label='Alonso')
plt.plot(media_anual_sainz['year'], media_anual_sainz['diferencia_posiciones'], marker='s', color='blue', label='Sainz')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Año')
plt.ylabel('Promedio de posiciones ganadas/perdidas')
plt.title('Evolución de la media de posiciones ganadas/perdidas - Alonso vs Sainz (carreras terminadas)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/14_grafica.png")

# plt.show()
##########################GRAFICA 15
# Calculamos la media de diferencia de posiciones por circuito para Alonso (solo carreras terminadas)
media_circuito_alonso = df_alonso_terminadas.groupby('name')['diferencia_posiciones'].mean().reset_index()

# Ordenamos por la media (de mayor a menor)
media_circuito_alonso = media_circuito_alonso.sort_values(by='diferencia_posiciones', ascending=False)

# Gráfico de barras con los circuitos donde más mejora posiciones Alonso
plt.figure(figsize=(10, 6))
plt.barh(media_circuito_alonso['name'], media_circuito_alonso['diferencia_posiciones'], color='teal')
plt.xlabel('Promedio de posiciones ganadas/perdidas')
plt.title('Circuitos favoritos de Alonso (solo carreras terminadas)')
plt.gca().invert_yaxis()  # Ponemos el circuito con mejor promedio arriba
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/15_grafica.png")

# plt.show()
##########################GRAFICA 15
# Calculamos la media de diferencia de posiciones por circuito para Sainz (solo carreras terminadas)
media_circuito_sainz = df_sainz_terminadas.groupby('name')['diferencia_posiciones'].mean().reset_index()

# Ordenamos por la media (de mayor a menor)
media_circuito_sainz = media_circuito_sainz.sort_values(by='diferencia_posiciones', ascending=False)

# Gráfico de barras con los circuitos donde más mejora posiciones Sainz
plt.figure(figsize=(10, 6))
plt.barh(media_circuito_sainz['name'], media_circuito_sainz['diferencia_posiciones'], color='lightblue')
plt.xlabel('Promedio de posiciones ganadas/perdidas')
plt.title('Circuitos favoritos de Sainz (solo carreras terminadas)')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/16_grafica.png")

# plt.show()
##########################GRAFICA 16
# Combinamos los circuitos de Alonso y Sainz
# Primero, renombramos las columnas para que no se solapen al hacer merge
media_circuito_alonso_renamed = media_circuito_alonso.rename(columns={'diferencia_posiciones': 'Alonso'})
media_circuito_sainz_renamed = media_circuito_sainz.rename(columns={'diferencia_posiciones': 'Sainz'})

# Hacemos merge de ambos dataframes por el nombre del circuito
df_comparacion_circuitos = pd.merge(
    media_circuito_alonso_renamed[['name', 'Alonso']],
    media_circuito_sainz_renamed[['name', 'Sainz']],
    on='name',
    how='outer'  # Incluimos todos los circuitos de ambos pilotos
)

# Rellenamos los NaN con 0 para indicar que no hay datos en ese circuito para ese piloto
df_comparacion_circuitos = df_comparacion_circuitos.fillna(0)

# Ordenamos por la suma total de diferencia de posiciones para tener los circuitos "más activos" arriba
df_comparacion_circuitos['suma'] = df_comparacion_circuitos['Alonso'] + df_comparacion_circuitos['Sainz']
df_comparacion_circuitos = df_comparacion_circuitos.sort_values(by='suma', ascending=False)

# Gráfico comparativo: barras agrupadas
plt.figure(figsize=(12, 8))
bar_width = 0.4
index = range(len(df_comparacion_circuitos))

plt.barh([i - bar_width/2 for i in index], df_comparacion_circuitos['Alonso'], height=bar_width, label='Alonso', color='orange')
plt.barh([i + bar_width/2 for i in index], df_comparacion_circuitos['Sainz'], height=bar_width, label='Sainz', color='blue')

plt.yticks(index, df_comparacion_circuitos['name'])
plt.xlabel('Promedio de posiciones ganadas/perdidas')
plt.title('Comparativa de circuitos favoritos - Alonso vs Sainz')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/17_grafica.png")

# plt.show()

##########################GRAFICA 16
# Reordenamos usando el nombre correcto de la columna: 'diferencia_posiciones'
ranking_circuitos_alonso = media_circuito_alonso.sort_values(by='diferencia_posiciones', ascending=False).reset_index(drop=True)

# Creamos la gráfica de barras con los circuitos más favorables para Alonso
plt.figure(figsize=(10, 6))
plt.barh(ranking_circuitos_alonso['name'][:10], ranking_circuitos_alonso['diferencia_posiciones'][:10], color='green')
plt.xlabel('Promedio de posiciones ganadas/perdidas')
plt.title('Top 10 circuitos más favorables para Alonso')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/18_grafica.png")

# plt.show()
##########################GRAFICA 17
# Ordenamos por la media de posiciones ganadas (mayor a menor) para Sainz
ranking_circuitos_sainz = media_circuito_sainz.sort_values(by='diferencia_posiciones', ascending=False).reset_index(drop=True)

# Gráfico de barras con los circuitos más favorables para Sainz
plt.figure(figsize=(10, 6))
plt.barh(ranking_circuitos_sainz['name'][:10], ranking_circuitos_sainz['diferencia_posiciones'][:10], color='skyblue')
plt.xlabel('Promedio de posiciones ganadas/perdidas')
plt.title('Top 10 circuitos más favorables para Sainz')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/19_grafica.png")

# plt.show()
##########################GRAFICA18
# Combinamos los circuitos de Alonso y Sainz
# Primero, renombramos las columnas para que no se solapen al hacer merge
media_circuito_alonso_renamed = media_circuito_alonso.rename(columns={'diferencia_posiciones': 'Alonso'})
media_circuito_sainz_renamed = media_circuito_sainz.rename(columns={'diferencia_posiciones': 'Sainz'})

# Hacemos merge de ambos dataframes por el nombre del circuito
df_comparacion_circuitos = pd.merge(
    media_circuito_alonso_renamed[['name', 'Alonso']],
    media_circuito_sainz_renamed[['name', 'Sainz']],
    on='name',
    how='outer'  # Incluimos todos los circuitos de ambos pilotos
)

# Rellenamos los NaN con 0 para indicar que no hay datos en ese circuito para ese piloto
df_comparacion_circuitos = df_comparacion_circuitos.fillna(0)

# Ordenamos por la suma total de diferencia de posiciones para tener los circuitos "más activos" arriba
df_comparacion_circuitos['suma'] = df_comparacion_circuitos['Alonso'] + df_comparacion_circuitos['Sainz']
df_comparacion_circuitos = df_comparacion_circuitos.sort_values(by='suma', ascending=False)
##########################GRAFICA19
# Gráfico comparativo: barras agrupadas
plt.figure(figsize=(12, 8))
bar_width = 0.4
index = range(len(df_comparacion_circuitos))

plt.barh([i - bar_width/2 for i in index], df_comparacion_circuitos['Alonso'], height=bar_width, label='Alonso', color='orange')
plt.barh([i + bar_width/2 for i in index], df_comparacion_circuitos['Sainz'], height=bar_width, label='Sainz', color='blue')

plt.yticks(index, df_comparacion_circuitos['name'])
plt.xlabel('Promedio de posiciones ganadas/perdidas')
plt.title('Comparativa de circuitos favoritos - Alonso vs Sainz')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/20_grafica.png")

# plt.show()
##########################GRAFICA20

# Reordenamos usando el nombre correcto de la columna: 'diferencia_posiciones'
ranking_circuitos_alonso = media_circuito_alonso.sort_values(by='diferencia_posiciones', ascending=False).reset_index(drop=True)

# Creamos la gráfica de barras con los circuitos más favorables para Alonso
plt.figure(figsize=(10, 6))
plt.barh(ranking_circuitos_alonso['name'][:10], ranking_circuitos_alonso['diferencia_posiciones'][:10], color='green')
plt.xlabel('Promedio de posiciones ganadas/perdidas')
plt.title('Top 10 circuitos más favorables para Alonso')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/21_grafica.png")

# plt.show()
##########################GRAFICA21
# Ordenamos por la media de posiciones ganadas (mayor a menor) para Sainz
ranking_circuitos_sainz = media_circuito_sainz.sort_values(by='diferencia_posiciones', ascending=False).reset_index(drop=True)

# Gráfico de barras con los circuitos más favorables para Sainz
plt.figure(figsize=(10, 6))
plt.barh(ranking_circuitos_sainz['name'][:10], ranking_circuitos_sainz['diferencia_posiciones'][:10], color='skyblue')
plt.xlabel('Promedio de posiciones ganadas/perdidas')
plt.title('Top 10 circuitos más favorables para Sainz')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/22_grafica.png")

# plt.show()
##########################GRAFICA22
#######################################################################################

# Dividimos en dos etapas: 2001-2010 y 2011-2024 (solo carreras terminadas)
# etapa_1 = df_alonso_terminadas[(df_alonso_terminadas['year'] >= 2001) & (df_alonso_terminadas['year'] <= 2010)]
# etapa_2 = df_alonso_terminadas[(df_alonso_terminadas['year'] >= 2011) & (df_alonso_terminadas['year'] <= 2024)]

# Calculamos la media de posiciones ganadas/perdidas por etapa
# media_etapa_1 = etapa_1['diferencia_posiciones'].mean()
# media_etapa_2 = etapa_2['diferencia_posiciones'].mean()

# Sacamos también los constructores con los que compitió en cada etapa
# constructores_etapa_1 = etapa_1['constructor_name'].unique()
# constructores_etapa_2 = etapa_2['constructor_name'].unique()


#constructores_etapa_1 = etapa_1['diferencia_posiciones'].unique()
#constructores_etapa_2 = etapa_2['diferencia_posiciones'].unique()
#######################################################################################
# Reconstituimos el dataframe con toda la información para Alonso (solo carreras terminadas)
etapa_1 = df_pilotos_completo[
    (df_pilotos_completo['surname'] == 'Alonso') &
    (df_pilotos_completo['year'] >= 2001) &
    (df_pilotos_completo['year'] <= 2010) &
    (~df_pilotos_completo['position'].isna())
][['driverId', 'surname', 'raceId', 'name', 'year', 'grid', 'position', 'constructor_name']].copy()

etapa_1['grid'] = pd.to_numeric(etapa_1['grid'], errors='coerce')
etapa_1['position'] = pd.to_numeric(etapa_1['position'], errors='coerce')
etapa_1['diferencia_posiciones'] = etapa_1['grid'] - etapa_1['position']

# Igual para etapa 2
etapa_2 = df_pilotos_completo[
    (df_pilotos_completo['surname'] == 'Alonso') &
    (df_pilotos_completo['year'] >= 2011) &
    (df_pilotos_completo['year'] <= 2024) &
    (~df_pilotos_completo['position'].isna())
][['driverId', 'surname', 'raceId', 'name', 'year', 'grid', 'position', 'constructor_name']].copy()

etapa_2['grid'] = pd.to_numeric(etapa_2['grid'], errors='coerce')
etapa_2['position'] = pd.to_numeric(etapa_2['position'], errors='coerce')
etapa_2['diferencia_posiciones'] = etapa_2['grid'] - etapa_2['position']
media_etapa_1 = etapa_1['diferencia_posiciones'].mean()
media_etapa_2 = etapa_2['diferencia_posiciones'].mean()
# Ahora ya existen las columnas necesarias en cada etapa y puedes hacer:
constructores_etapa_1 = etapa_1['constructor_name'].unique()
constructores_etapa_2 = etapa_2['constructor_name'].unique()

print("Constructores etapa 1:", constructores_etapa_1)
print("Constructores etapa 2:", constructores_etapa_2)


# Preparamos gráfico comparativo
plt.figure(figsize=(10, 6))
bars = plt.bar(['2001-2010', '2011-2024'], [media_etapa_1, media_etapa_2], color=['orange', 'green'])
plt.ylabel('Promedio de posiciones ganadas/perdidas')
plt.title('Comparación de etapas - Alonso')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Añadimos los constructores como texto dentro de las barras y más abajo
etiqueta_etapa_1 = ', '.join(constructores_etapa_1)
etiqueta_etapa_2 = ', '.join(constructores_etapa_2)

# Ajustamos las posiciones de texto para que estén dentro de las barras (más abajo)
for i, etiqueta in enumerate([etiqueta_etapa_1, etiqueta_etapa_2]):
    plt.text(i, 0.05, f"Constructores:\n{etiqueta}", ha='center', va='bottom', fontsize=9, color='white')

plt.tight_layout()
plt.savefig("../img/Graficas_generadas/23_grafica.png")

# plt.show()
##########################GRAFICA

# Primero, reconstruimos df_alonso_terminadas incluyendo constructor_name  
df_alonso_terminadas = df_espanoles_mas_5_carreras[
    (df_espanoles_mas_5_carreras['driverId'] == 4) &
    (~df_espanoles_mas_5_carreras['position'].isna())
][['driverId', 'surname', 'raceId', 'name', 'year', 'grid', 'position', 'constructor_name']]

df_alonso_terminadas['diferencia_posiciones'] = df_alonso_terminadas['grid'] - df_alonso_terminadas['position']

# Hacemos lo mismo para Sainz
df_sainz_terminadas = df_espanoles_mas_5_carreras[
    (df_espanoles_mas_5_carreras['driverId'] == driver_id_sainz) &
    (~df_espanoles_mas_5_carreras['position'].isna())
][['driverId', 'surname', 'raceId', 'name', 'year', 'grid', 'position', 'constructor_name']]

df_sainz_terminadas['diferencia_posiciones'] = df_sainz_terminadas['grid'] - df_sainz_terminadas['position']

# Ahora recalculamos la media de diferencia de posiciones por constructor para Alonso
media_constructor_alonso = df_alonso_terminadas.groupby('constructor_name')['diferencia_posiciones'].mean().reset_index()
media_constructor_alonso = media_constructor_alonso.rename(columns={'diferencia_posiciones': 'Alonso'})

# Recalculamos para Sainz
media_constructor_sainz = df_sainz_terminadas.groupby('constructor_name')['diferencia_posiciones'].mean().reset_index()
media_constructor_sainz = media_constructor_sainz.rename(columns={'diferencia_posiciones': 'Sainz'})

# Combinamos ambos dataframes por constructor
df_comparacion_constructores = pd.merge(
    media_constructor_alonso[['constructor_name', 'Alonso']],
    media_constructor_sainz[['constructor_name', 'Sainz']],
    on='constructor_name',
    how='outer'
).fillna(0)
##########################GRAFICA23
# Ordenamos por la suma total para ver los más destacados arriba
df_comparacion_constructores['suma'] = df_comparacion_constructores['Alonso'] + df_comparacion_constructores['Sainz']
df_comparacion_constructores = df_comparacion_constructores.sort_values(by='suma', ascending=False)

# Gráfico comparativo: barras agrupadas
plt.figure(figsize=(10, 6))
bar_width = 0.4
index = range(len(df_comparacion_constructores))

plt.barh([i - bar_width/2 for i in index], df_comparacion_constructores['Alonso'], height=bar_width, label='Alonso', color='orange')
plt.barh([i + bar_width/2 for i in index], df_comparacion_constructores['Sainz'], height=bar_width, label='Sainz', color='blue')

plt.yticks(index, df_comparacion_constructores['constructor_name'])
plt.xlabel('Promedio de posiciones ganadas/perdidas')
plt.title('Comparativa por constructor - Alonso vs Sainz')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/24_grafica.png")

# plt.show()

##########################GRAFICA24
# Reconstruimos el dataframe general, pero ahora para TODOS los pilotos (no solo españoles)

# Paso 1: drivers - información general del piloto
drivers_df = drivers[['driverId', 'driverRef', 'forename', 'surname', 'nationality']]

# Paso 2: results - datos finales de la carrera
results_df = results[['driverId', 'raceId', 'constructorId', 'grid', 'position', 'points', 'statusId']]

# Paso 3: qualifying - posición de clasificación
qualifying = pd.read_csv('../Datos/Datos_csv/qualifying.csv')
qualifying_df = qualifying[['driverId', 'raceId', 'position']].rename(columns={'position': 'qualifying_position'})

# Paso 4: races - información de la carrera
races_df = races[['raceId', 'year', 'round', 'circuitId', 'name', 'date']]

# Paso 5: circuits - ubicación y país del circuito
circuits = pd.read_csv('../Datos/Datos_csv/circuits.csv')
circuits_df = circuits[['circuitId', 'name', 'location', 'country']].rename(columns={'name': 'circuit_name'})

# Paso 6: constructors - información del constructor
constructors_df = constructors[['constructorId', 'name', 'nationality']].rename(columns={'name': 'constructor_name', 'nationality': 'constructor_nationality'})

# Paso 7: años en los que cada constructor participó
constructor_standings = pd.read_csv('../Datos/Datos_csv/constructor_standings.csv')
years_constructor = constructor_standings[['constructorId', 'raceId']].merge(races[['raceId', 'year']], on='raceId', how='left')
years_by_constructor = years_constructor.groupby('constructorId')['year'].unique().reset_index().rename(columns={'year': 'years_competed'})

# Paso 8: status - descripción del resultado final
status = pd.read_csv('../Datos/Datos_csv/status.csv')
status_df = status[['statusId', 'status']]

# Ahora realizamos los merges progresivos
df_pilotos = drivers_df.merge(results_df, on='driverId', how='left')
df_pilotos = df_pilotos.merge(qualifying_df, on=['driverId', 'raceId'], how='left')
df_pilotos = df_pilotos.merge(races_df, on='raceId', how='left')
df_pilotos = df_pilotos.merge(circuits_df, on='circuitId', how='left')
df_pilotos = df_pilotos.merge(constructors_df, on='constructorId', how='left')
df_pilotos = df_pilotos.merge(years_by_constructor, on='constructorId', how='left')
df_pilotos = df_pilotos.merge(status_df, on='statusId', how='left')


# Recalculamos carreras totales y puntos totales/promedio
carreras_por_piloto = results.groupby('driverId').size().reset_index(name='num_carreras')
puntos_por_piloto = results.groupby('driverId')['points'].agg(['sum', 'mean']).reset_index().rename(columns={'sum': 'total_puntos', 'mean': 'puntos_promedio'})

# Creamos el dataframe final con la info básica del piloto y los resultados agregados
df_pilotos_final = drivers[['driverId', 'surname', 'nationality']].merge(
    carreras_por_piloto, on='driverId', how='left').merge(
    puntos_por_piloto, on='driverId', how='left')

# Ahora seleccionamos los 2 mejores pilotos por nacionalidad (10 nacionalidades más fuertes)
puntos_por_nacionalidad = df_pilotos_final.groupby('nationality')['total_puntos'].sum().reset_index().sort_values(by='total_puntos', ascending=False)
nacionalidades_top10 = puntos_por_nacionalidad.head(10)['nationality'].tolist()

# Seleccionamos 2 pilotos por cada nacionalidad top
pilotos_destacados_final = pd.DataFrame()
for nacionalidad in nacionalidades_top10:
    top_pilotos = df_pilotos_final[df_pilotos_final['nationality'] == nacionalidad].sort_values(by='total_puntos', ascending=False).head(2)
    pilotos_destacados_final = pd.concat([pilotos_destacados_final, top_pilotos])

# Añadimos a Alonso explícitamente en caso de que no aparezca en el top
alonso_data = df_pilotos_final[df_pilotos_final['surname'] == 'Alonso']
pilotos_destacados_final = pd.concat([pilotos_destacados_final, alonso_data]).drop_duplicates(subset=['driverId']).reset_index(drop=True)

##########################GRAFICA25
# Gráfico con la leyenda incluyendo la media de puntos por carrera y el número de carreras

# Ajustamos la escala para que se vea mejor y añadimos los puntos totales al final de cada barra

plt.figure(figsize=(12, 6))
bars = plt.barh(pilotos_destacados_final['surname'], pilotos_destacados_final['total_puntos'], color='teal')
plt.xlabel('Puntos totales')
plt.title('Puntos totales - Alonso vs pilotos destacados (puntos totales y escala ajustada)')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Añadimos los puntos totales como texto al final de cada barra (ajustamos para que no sobresalga mucho)
for bar, puntos in zip(bars, pilotos_destacados_final['total_puntos']):
    width = bar.get_width()
    plt.text(width + 50, bar.get_y() + bar.get_height()/2, f"{int(puntos)}", va='center', fontsize=9, color='black')

# Ajustamos la escala de los ejes para que las barras no sean demasiado largas visualmente
plt.xlim(0, pilotos_destacados_final['total_puntos'].max() * 1.1)

# Leyenda con más información (ya estaba bien)
legend_labels = [f"{surname} ({nat}) - Media: {prom:.1f} - Carreras: {carreras}" 
                  for surname, nat, prom, carreras in zip(pilotos_destacados_final['surname'], 
                                                           pilotos_destacados_final['nationality'], 
                                                           pilotos_destacados_final['puntos_promedio'], 
                                                           pilotos_destacados_final['num_carreras'])]
plt.legend(bars, legend_labels, title='Nacionalidad, Media y Carreras', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

plt.tight_layout()
plt.savefig("../img/Graficas_generadas/25_grafica.png")

# plt.show()

##########################GRAFICA26
# Calculamos puntos totales por nacionalidad (sumando todos los años)
# Partimos de pilotos_destacados_final (con toda la info de cada carrera, piloto, nacionalidad, año y puntos)


# Aseguramos que los puntos son numéricos
df_pilotos['points'] = pd.to_numeric(df_pilotos['points'], errors='coerce')

# Calculamos los puntos totales por año y nacionalidad
puntos_por_anyo_nacionalidad = df_pilotos.groupby(['year', 'nationality'])['points'].sum().reset_index()

# Calculamos puntos totales por nacionalidad
puntos_totales_nacionalidad = puntos_por_anyo_nacionalidad.groupby('nationality')['points'].sum().reset_index()

# Filtramos nacionalidades con más de 1000 puntos
nacionalidades_fuertes = puntos_totales_nacionalidad[puntos_totales_nacionalidad['points'] >= 1000]['nationality'].tolist()

# Filtramos el dataframe con estas nacionalidades
puntos_por_anyo_nacionalidad_filtrado = puntos_por_anyo_nacionalidad[puntos_por_anyo_nacionalidad['nationality'].isin(nacionalidades_fuertes)]

# Gráfica final
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for nacionalidad in nacionalidades_fuertes:
    data_nac = puntos_por_anyo_nacionalidad_filtrado[puntos_por_anyo_nacionalidad_filtrado['nationality'] == nacionalidad]
    plt.plot(data_nac['year'], data_nac['points'], marker='o', label=nacionalidad)

plt.xlabel('Año')
plt.ylabel('Puntos totales')
plt.title('Evolución de puntos totales por año (nacionalidades con >1000 puntos)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("../img/Graficas_generadas/26_grafica.png")

#plt.show()
##########################GRAFICA27
#################################################
#longevidad y competitividad

# 4️⃣ Filtramos solo datos desde 1990
df_moderna = df_pilotos_completo[df_pilotos_completo['year'] >= 1990]

# 5️⃣ Calculamos años únicos activos por piloto
años_activos = df_moderna.groupby('surname')['year'].nunique().reset_index().rename(columns={'year': 'años_actividad'})

# 6️⃣ Calculamos puntos totales y promedio por carrera por piloto
puntos_stats = df_moderna.groupby('surname')['points'].agg(['sum', 'mean']).reset_index().rename(columns={'sum': 'puntos_totales', 'mean': 'puntos_promedio'})

# 7️⃣ Combinamos en un solo dataframe
df_longevidad = años_activos.merge(puntos_stats, on='surname', how='left').sort_values(by='años_actividad', ascending=False).reset_index(drop=True)

df_longevidad

df_moderna = df_pilotos_completo[df_pilotos_completo['year'] >= 1990]
años_activos = df_moderna.groupby('surname')['year'].nunique().reset_index().rename(columns={'year': 'años_actividad'})
puntos_stats = df_moderna.groupby('surname')['points'].agg(['sum', 'mean']).reset_index().rename(columns={'sum': 'puntos_totales', 'mean': 'puntos_promedio'})

# Combinamos y filtramos
df_longevidad = años_activos.merge(puntos_stats, on='surname', how='left')
df_longevidad_filtrado = df_longevidad[(df_longevidad['años_actividad'] >= 1) & (df_longevidad['puntos_totales'] >= 100)].sort_values(by='años_actividad', ascending=False).reset_index(drop=True)

df_longevidad_filtrado.to_csv('../Datos/Datos_csv_creados/_df_longevidad_filtrado.csv', index=True)

##########################GRAFICA28
# grafica 1
df_plot = df_longevidad_filtrado.sort_values(by='años_actividad', ascending=True)
# Ordenamos la leyenda de mayor a menor por puntos totales
orden_leyenda = df_plot.sort_values(by='puntos_totales', ascending=False)

plt.figure(figsize=(10, 8))
bars = plt.barh(df_plot['surname'], df_plot['años_actividad'], color='skyblue')
plt.xlabel('Años de actividad')
plt.ylabel('Piloto')
plt.title('Años de actividad por piloto (1990 en adelante)')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Leyenda ordenada por puntos totales
legend_labels = [f"{surname}: {int(puntos)} puntos totales"
                  for surname, puntos in zip(orden_leyenda['surname'], orden_leyenda['puntos_totales'])]
plt.legend(bars, legend_labels, title='Puntos totales', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

plt.tight_layout()
plt.savefig("../img/Graficas_generadas/27_grafica.png")

plt.show()

# grafica 2

# Ordenamos la leyenda de mayor a menor por puntos totales
#orden_leyenda = df_plot.sort_values(by='puntos_totales', ascending=False)

#plt.figure(figsize=(10, 8))
#bars = plt.barh(df_plot['surname'], df_plot['años_actividad'], color='skyblue')
#plt.xlabel('Años de actividad')
#plt.ylabel('Piloto')
#plt.title('Años de actividad por piloto (1990 en adelante)')
#plt.grid(axis='x', linestyle='--', alpha=0.7)

# Leyenda ordenada por puntos totales
#legend_labels = [f"{surname}: {int(puntos)} puntos totales"
#                  for surname, puntos in zip(orden_leyenda['surname'], orden_leyenda['puntos_totales'])]
#plt.legend(bars, legend_labels, title='Puntos totales', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

#plt.tight_layout()
#plt.savefig("../img/Graficas_generadas/28_grafica.png")

#plt.show()
##########################GRAFICA2
# Creamos un scatterplot con colores únicos para cada piloto y leyenda con puntos totales

plt.figure(figsize=(10, 6))

# Generamos colores únicos para cada piloto
colors = plt.cm.tab20(range(len(df_longevidad_filtrado)))

# Scatterplot con cada piloto en color distinto
for i, row in df_longevidad_filtrado.iterrows():
    plt.scatter(row['años_actividad'], row['puntos_totales'], color=colors[i], label=f"{row['surname']}: {int(row['puntos_totales'])} puntos")

plt.xlabel('Años de actividad')
plt.ylabel('Puntos totales')
plt.title('Relación entre longevidad y puntos totales (con leyenda)')
plt.grid(True, linestyle='--', alpha=0.7)

# Leyenda externa con puntos totales
plt.legend(title='Piloto y puntos totales', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

plt.tight_layout()
#plt.show()
#grafica 3
# 3️⃣ Gráfico combinado de barras (años de actividad) y línea (puntos promedio)
##########################GRAFICA32
# Ordenamos por años de actividad para la visualización combinada
df_plot = df_longevidad_filtrado.sort_values(by='años_actividad', ascending=True)

fig, ax1 = plt.subplots(figsize=(12, 6))

# Barras para los años de actividad
ax1.bar(df_plot['surname'], df_plot['años_actividad'], color='skyblue', label='Años de actividad')
ax1.set_ylabel('Años de actividad', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.set_xlabel('Piloto')
ax1.set_xticklabels(df_plot['surname'], rotation=45, ha='right')

# Línea para el promedio de puntos por carrera
ax2 = ax1.twinx()
ax2.plot(df_plot['surname'], df_plot['puntos_promedio'], color='darkgreen', marker='o', label='Promedio de puntos')
ax2.set_ylabel('Promedio de puntos por carrera', color='darkgreen')
ax2.tick_params(axis='y', labelcolor='darkgreen')

# Título y leyenda combinada
fig.suptitle('Años de actividad y rendimiento competitivo por piloto', fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.9)  # Ajuste para título superior

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("../img/Graficas_generadas/29_grafica.png")

#plt.show()