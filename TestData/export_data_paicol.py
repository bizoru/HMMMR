# Evaluacion
## Dataset
#Explicar
### Estaciones
#### Objetivo
#'21057060': PAICOL
target = '21087080-WL_CAL_AVG'
#### Predictoras
# PAICOL
preds_cod = ['21017060', '21017040' ,'21087080']
# Removed PTE balseadero (Data hasta el 2015) 21047010
### Variables
#--NOT-- PR_CAL_ACU -> Precipitacion acumulada horaria
#WL_CAL_AVG -> Nivel promedio horario
#PR_CAL_ITS -> Intensidad de Precipitacion horaria
siglas_vars =  ["WL_CAL_AVG", "PR_CAL_ITS"]
#### Lags
from emjav.emjav_data.tools import cargar_valores_observados
from datetime import datetime, timedelta
import pandas as pd
from pandas import Series
import numpy as np
fecha_inicial = datetime(2015, 1, 1, 0, 0)
fecha_final = datetime(2018, 1, 1, 0, 0)
max_lag = 10
series = {}
base_keys = []
for est_code in preds_cod:
    for sigla in siglas_vars:
        key = "{}-{}".format(est_code, sigla)
        base_keys.append(key)
        series[key] = cargar_valores_observados(est_code, sigla, fecha_inicial, fecha_final, return_type="Series")

for b_key in base_keys:
    for i in range(1,10):
        # First shifts
        series["{}-Lag{}".format(b_key,i)] = series[b_key].shift(i)
    if b_key != target:
        del series[b_key]


# Then dropna
predictor_data  = pd.DataFrame(series)
cleaned_data = predictor_data.dropna()
# Reorder columns
ordered_columns = list(cleaned_data.columns)
target_col_index = cleaned_data.columns.get_loc(target)
ordered_columns[target_col_index] = ordered_columns[-1] # Replace  by last existing column
ordered_columns.pop() # And removed not used!
cleaned_data['Ones'] = Series(np.ones(cleaned_data.shape[0]), index=cleaned_data.index)
ordered_data = cleaned_data[ordered_columns+['Ones',target]].sort_index(ascending=False) # Recent data is most relevant
data_window = 300
ordered_data = ordered_data[:data_window]
ordered_data.to_csv("/tmp/pronos_ordered_cleaned.csv", index=False)
