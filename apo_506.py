#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversión de notebook R a Python
APO-506: LightGBM con Canaritos
"""

import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import gc
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

print(f"Inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Limpio memoria
gc.collect()

# Parámetros locales
plocal = {
    'qcanaritos': 5,
    'min_data_in_leaf': 2000,
    'learning_rate': 1.0,
    'gradient_bound': 0.01,
    'APO': 1,
    'ksemillerio': 1
}

# Parámetros generales
PARAM = {
    'experimento': 'apo-506',
    'semilla_primigenia': 102191
}

# Crear estructura de directorios
exp_folder = f"/content/buckets/b1/exp/{PARAM['experimento']}"
os.makedirs(exp_folder, exist_ok=True)
os.chdir(exp_folder)

# ============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

print(f"Cargando dataset: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Leo el dataset
dataset = pd.read_csv("~/datasets/competencia_02_crudo.csv.gz", compression='gzip')

# Calculo el periodo0 consecutivo
dataset['periodo0'] = (dataset['foto_mes'] // 100) * 12 + (dataset['foto_mes'] % 100)

# Ordeno por cliente y periodo
dataset = dataset.sort_values(['numero_de_cliente', 'periodo0']).reset_index(drop=True)

# Calculo topes
periodo_ultimo = dataset['periodo0'].max()
periodo_anteultimo = periodo_ultimo - 1

# Calculo los leads de orden 1 y 2
dataset['periodo1'] = dataset.groupby('numero_de_cliente')['periodo0'].shift(-1)
dataset['periodo2'] = dataset.groupby('numero_de_cliente')['periodo0'].shift(-2)

# Inicializo clase_ternaria
dataset['clase_ternaria'] = None

# Assign most common class values = "CONTINUA"
mask = dataset['periodo0'] < periodo_anteultimo
dataset.loc[mask, 'clase_ternaria'] = 'CONTINUA'

# Calculo BAJA+1
mask = (dataset['periodo0'] < periodo_ultimo) & \
       ((dataset['periodo1'].isna()) | (dataset['periodo0'] + 1 < dataset['periodo1']))
dataset.loc[mask, 'clase_ternaria'] = 'BAJA+1'

# Calculo BAJA+2
mask = (dataset['periodo0'] < periodo_anteultimo) & \
       (dataset['periodo0'] + 1 == dataset['periodo1']) & \
       ((dataset['periodo2'].isna()) | (dataset['periodo0'] + 2 < dataset['periodo2']))
dataset.loc[mask, 'clase_ternaria'] = 'BAJA+2'

# Limpio columnas auxiliares
dataset = dataset.drop(columns=['periodo1', 'periodo2'])

print(f"Dataset cargado. Shape: {dataset.shape}")
gc.collect()

# ============================================================================
# FEATURE ENGINEERING BÁSICO
# ============================================================================

print(f"Feature Engineering básico: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Salsa Magica para 202106
if 'mprestamos_personales' in dataset.columns:
    dataset = dataset.drop(columns=['mprestamos_personales'])
if 'cprestamos_personales' in dataset.columns:
    dataset = dataset.drop(columns=['cprestamos_personales'])

# El mes 1,2,..12, podría servir para detectar estacionalidad
dataset['kmes'] = dataset['foto_mes'] % 100

# Creo un ctr_quarter normalizado
dataset['ctrx_quarter_normalizado'] = dataset['ctrx_quarter'].astype(float)
dataset.loc[dataset['cliente_antiguedad'] == 1, 'ctrx_quarter_normalizado'] = \
    dataset.loc[dataset['cliente_antiguedad'] == 1, 'ctrx_quarter'] * 5.0
dataset.loc[dataset['cliente_antiguedad'] == 2, 'ctrx_quarter_normalizado'] = \
    dataset.loc[dataset['cliente_antiguedad'] == 2, 'ctrx_quarter'] * 2.0
dataset.loc[dataset['cliente_antiguedad'] == 3, 'ctrx_quarter_normalizado'] = \
    dataset.loc[dataset['cliente_antiguedad'] == 3, 'ctrx_quarter'] * 1.2

# Variable extraída de una tesis de maestría de Irlanda
dataset['mpayroll_sobre_edad'] = dataset['mpayroll'] / dataset['cliente_edad']

print(f"FE básico completado: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# FUNCIÓN PARA TENDENCIAS (simplificada para Python)
# ============================================================================

def TendenciaYmuchomas(dataset, cols, ventana=6, tendencia=True, 
                       minimo=True, maximo=True, promedio=True,
                       ratioavg=False, ratiomax=False):
    """
    Calcula tendencias, mínimos, máximos y promedios de los últimos N meses
    Versión simplificada para Python
    """
    print(f"  Calculando tendencias con ventana={ventana}")
    
    for col in cols:
        if col not in dataset.columns:
            continue
            
        # Calculo rolling statistics por cliente
        grouped = dataset.groupby('numero_de_cliente')[col]
        
        if tendencia:
            # Tendencia como diferencia simple (simplificado)
            dataset[f'{col}_tend{ventana}'] = grouped.diff(ventana)
        
        if minimo:
            dataset[f'{col}_min{ventana}'] = grouped.transform(
                lambda x: x.rolling(window=ventana, min_periods=1).min()
            )
        
        if maximo:
            dataset[f'{col}_max{ventana}'] = grouped.transform(
                lambda x: x.rolling(window=ventana, min_periods=1).max()
            )
        
        if promedio:
            avg_col = grouped.transform(
                lambda x: x.rolling(window=ventana, min_periods=1).mean()
            )
            dataset[f'{col}_avg{ventana}'] = avg_col
            
            if ratioavg:
                dataset[f'{col}_ratioavg{ventana}'] = dataset[col] / avg_col
        
        if ratiomax and maximo:
            max_col = dataset[f'{col}_max{ventana}']
            dataset[f'{col}_ratiomax{ventana}'] = dataset[col] / max_col

# ============================================================================
# FEATURE ENGINEERING HISTÓRICO - LAGS
# ============================================================================

print(f"Creando LAGs: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Ordeno por cliente y mes
dataset = dataset.sort_values(['numero_de_cliente', 'foto_mes']).reset_index(drop=True)

# Columnas lagueables
cols_lagueables = [col for col in dataset.columns 
                   if col not in ['numero_de_cliente', 'foto_mes', 'clase_ternaria', 'periodo0']]

# Lags de orden 1 y 2
for col in cols_lagueables:
    if col in dataset.columns:
        dataset[f'{col}_lag1'] = dataset.groupby('numero_de_cliente')[col].shift(1)
        dataset[f'{col}_lag2'] = dataset.groupby('numero_de_cliente')[col].shift(2)

# Agrego los delta lags
for col in cols_lagueables:
    if col in dataset.columns and f'{col}_lag1' in dataset.columns:
        dataset[f'{col}_delta1'] = dataset[col] - dataset[f'{col}_lag1']
        dataset[f'{col}_delta2'] = dataset[col] - dataset[f'{col}_lag2']

print(f"LAGs creados. Columnas: {len(dataset.columns)}")
gc.collect()

# ============================================================================
# FEATURE ENGINEERING HISTÓRICO - TENDENCIAS
# ============================================================================

# Parámetros de tendencias
PARAM['FE_hist'] = {
    'Tendencias': {
        'run': True,
        'ventana': 6,
        'tendencia': True,
        'minimo': False,
        'maximo': False,
        'promedio': False,
        'ratioavg': False,
        'ratiomax': False
    }
}

print(f"Calculando tendencias: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Actualizo cols_lagueables con las que existen
cols_lagueables = [col for col in cols_lagueables if col in dataset.columns]

# Ordeno nuevamente
dataset = dataset.sort_values(['numero_de_cliente', 'foto_mes']).reset_index(drop=True)

# Aplico tendencias si está habilitado
if PARAM['FE_hist']['Tendencias']['run']:
    TendenciaYmuchomas(
        dataset,
        cols=cols_lagueables,
        ventana=PARAM['FE_hist']['Tendencias']['ventana'],
        tendencia=PARAM['FE_hist']['Tendencias']['tendencia'],
        minimo=PARAM['FE_hist']['Tendencias']['minimo'],
        maximo=PARAM['FE_hist']['Tendencias']['maximo'],
        promedio=PARAM['FE_hist']['Tendencias']['promedio'],
        ratioavg=PARAM['FE_hist']['Tendencias']['ratioavg'],
        ratiomax=PARAM['FE_hist']['Tendencias']['ratiomax']
    )

print(f"Tendencias calculadas. Columnas totales: {len(dataset.columns)}")
print(f"FE completado: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# PREPARACIÓN PARA ENTRENAMIENTO
# ============================================================================

# Parámetros de entrenamiento final
PARAM['train_final'] = {
    'future': [202106],
    'training': [
        201901, 201902, 201903, 201904, 201905, 201906,
        201907, 201908, 201909, 201910, 201911, 201912,
        202001, 202002, 202003, 202004, 202005, 202006,
        202007, 202008, 202009, 202010, 202011, 202012,
        202101, 202102, 202103, 202104
    ],
    'undersampling': 0.05
}

# Filtro meses de entrenamiento
dataset_train_final = dataset[dataset['foto_mes'].isin(PARAM['train_final']['training'])].copy()

print(f"Dataset train_final creado. Shape: {dataset_train_final.shape}")

# ============================================================================
# CANARITOS
# ============================================================================

print(f"Agregando canaritos: {time.strftime('%Y-%m-%d %H:%M:%S')}")

PARAM['train_final']['lgbm'] = {'qcanaritos': plocal['qcanaritos']}

cols0 = dataset_train_final.columns.tolist()
filas = len(dataset_train_final)

if PARAM['train_final']['lgbm']['qcanaritos'] > 0:
    np.random.seed(PARAM['semilla_primigenia'])
    for i in range(1, PARAM['train_final']['lgbm']['qcanaritos'] + 1):
        dataset_train_final[f'canarito_{i}'] = np.random.uniform(0, 1, filas)
    
    # Las columnas canaritos van al comienzo
    cols_canaritos = [col for col in dataset_train_final.columns if col not in cols0]
    nuevas_columnas = cols_canaritos + cols0
    dataset_train_final = dataset_train_final[nuevas_columnas]

print(f"Canaritos agregados: {PARAM['train_final']['lgbm']['qcanaritos']}")

# ============================================================================
# UNDERSAMPLING
# ============================================================================

print(f"Aplicando undersampling: {time.strftime('%Y-%m-%d %H:%M:%S')}")

np.random.seed(PARAM['semilla_primigenia'])
dataset_train_final['azar'] = np.random.uniform(0, 1, len(dataset_train_final))
dataset_train_final['training'] = 0

mask = (dataset_train_final['azar'] <= PARAM['train_final']['undersampling']) | \
       (dataset_train_final['clase_ternaria'].isin(['BAJA+1', 'BAJA+2']))
dataset_train_final.loc[mask, 'training'] = 1

dataset_train_final = dataset_train_final.drop(columns=['azar'])

print(f"Registros para training: {dataset_train_final['training'].sum()}")

# ============================================================================
# CLASE BINARIA
# ============================================================================

# Paso la clase a binaria {0, 1}
dataset_train_final['clase01'] = dataset_train_final['clase_ternaria'].apply(
    lambda x: 1 if x in ['BAJA+2', 'BAJA+1'] else 0
)

# ============================================================================
# PARÁMETROS DE LIGHTGBM
# ============================================================================

PARAM['train_final']['lgbm']['param_completo'] = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'custom',
    'first_metric_only': False,
    'boost_from_average': True,
    'feature_pre_filter': False,
    'force_row_wise': True,
    'verbosity': -100,
    'seed': PARAM['semilla_primigenia'],
    'max_bin': 31,
    'min_data_in_leaf': plocal['min_data_in_leaf'],
    'num_iterations': 9999,
    'num_leaves': 9999,
    'learning_rate': plocal['learning_rate'],
    'feature_fraction': 0.50,
    # Parámetros específicos de zlightgbm (si se usa la versión modificada)
    # 'canaritos': PARAM['train_final']['lgbm']['qcanaritos'],
    # 'gradient_bound': plocal['gradient_bound']
}

# ============================================================================
# SEMILLERIO
# ============================================================================

print(f"Generando semillerio: {time.strftime('%Y-%m-%d %H:%M:%S')}")

PARAM['train_final']['APO'] = plocal['APO']
PARAM['train_final']['ksemillerio'] = plocal['ksemillerio']
PARAM['train_final']['cortes'] = [8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000]

# Genero semillas usando números primos
def generar_primos(minimo, maximo, cantidad):
    """Genera números primos en un rango"""
    primos = []
    for num in range(minimo, maximo):
        if num > 1:
            es_primo = True
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    es_primo = False
                    break
            if es_primo:
                primos.append(num)
                if len(primos) >= cantidad * 10:  # Genero extras
                    break
    return primos

# Genero primos y selecciono semillas
cantidad_semillas = PARAM['train_final']['APO'] * PARAM['train_final']['ksemillerio']
primos = generar_primos(100000, 1000000, cantidad_semillas)
np.random.seed(PARAM['semilla_primigenia'])
PARAM['train_final']['semillas'] = list(np.random.choice(primos, cantidad_semillas, replace=False))

print(f"Semillas generadas: {PARAM['train_final']['semillas']}")

# ============================================================================
# PREPARACIÓN DE DATOS PARA LIGHTGBM
# ============================================================================

# Campos buenos (excluyo variables auxiliares)
campos_buenos = [col for col in dataset_train_final.columns 
                 if col not in ['clase_ternaria', 'clase01', 'training', 'azar']]

print(f"Campos buenos: {len(campos_buenos)}")

# Preparo datos de entrenamiento
X_train = dataset_train_final[dataset_train_final['training'] == 1][campos_buenos]
y_train = dataset_train_final[dataset_train_final['training'] == 1]['clase01']

dtrain_final = lgb.Dataset(
    data=X_train,
    label=y_train,
    free_raw_data=False
)

print(f"Dataset LightGBM creado. Filas: {X_train.shape[0]}, Columnas: {X_train.shape[1]}")

# ============================================================================
# ENTRENAMIENTO DE MODELOS
# ============================================================================

print(f"Iniciando entrenamiento de modelos: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Creo directorio para modelitos
os.makedirs("modelitos", exist_ok=True)

param_completo = PARAM['train_final']['lgbm']['param_completo'].copy()

for sem in PARAM['train_final']['semillas']:
    arch_modelo = f"./modelitos/mod_{sem}.txt"
    
    if not os.path.exists(arch_modelo):
        print(f"  Entrenando modelo con semilla {sem}...")
        param_completo['seed'] = sem
        
        modelito = lgb.train(
            params=param_completo,
            train_set=dtrain_final
        )
        
        modelito.save_model(arch_modelo)
        del modelito
        gc.collect()

print(f"Modelos entrenados: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# PREPARACIÓN DE DATOS FUTURE
# ============================================================================

print(f"Preparando datos future: {time.strftime('%Y-%m-%d %H:%M:%S')}")

dfuture = dataset[dataset['foto_mes'].isin(PARAM['train_final']['future'])].copy()

cols0_future = dfuture.columns.tolist()
filas_future = len(dfuture)

# Agrego canaritos al future
if PARAM['train_final']['lgbm']['qcanaritos'] > 0:
    np.random.seed(PARAM['semilla_primigenia'] + 1)  # Semilla diferente para future
    for i in range(1, PARAM['train_final']['lgbm']['qcanaritos'] + 1):
        dfuture[f'canarito_{i}'] = np.random.uniform(0, 1, filas_future)
    
    # Las columnas canaritos van al comienzo
    cols_canaritos_future = [col for col in dfuture.columns if col not in cols0_future]
    nuevas_columnas_future = cols_canaritos_future + cols0_future
    dfuture = dfuture[nuevas_columnas_future]

# Preparo matriz de future
mfuture = dfuture[campos_buenos].values

# Calculo ganancia
dfuture['ganancia'] = dfuture['clase_ternaria'].apply(
    lambda x: 780000 if x == 'BAJA+2' else -20000
)

print(f"Datos future preparados. Shape: {dfuture.shape}")

# ============================================================================
# PREDICCIÓN Y EVALUACIÓN
# ============================================================================

print(f"Iniciando predicciones: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Matriz de ganancias
mganancias = np.zeros((PARAM['train_final']['APO'], len(PARAM['train_final']['cortes'])))

# Archivo de predicciones
if os.path.exists("prediccion.txt"):
    os.remove("prediccion.txt")

# Aplico el modelo a los datos del future
for vapo in range(PARAM['train_final']['APO']):
    print(f"  Procesando APO {vapo + 1}/{PARAM['train_final']['APO']}")
    
    # Inicialización en CERO
    vpred_acum = np.zeros(len(dfuture))
    qacumulados = 0
    
    desde = vapo * PARAM['train_final']['ksemillerio']
    hasta = desde + PARAM['train_final']['ksemillerio']
    semillas = PARAM['train_final']['semillas'][desde:hasta]
    
    for sem in semillas:
        arch_modelo = f"./modelitos/mod_{sem}.txt"
        
        if os.path.exists(arch_modelo):
            modelo_final = lgb.Booster(model_file=arch_modelo)
            vpred_acum += modelo_final.predict(mfuture)
            qacumulados += 1
            del modelo_final
            gc.collect()
    
    if qacumulados > 0:
        vpred_acum = vpred_acum / qacumulados  # Paso a probabilidad
        
        # Tabla de predicción
        tb_prediccion = dfuture[['numero_de_cliente', 'foto_mes', 'ganancia']].copy()
        tb_prediccion['meta_modelo'] = vapo + 1
        tb_prediccion['prob'] = vpred_acum
        tb_prediccion = tb_prediccion.sort_values('prob', ascending=False).reset_index(drop=True)
        tb_prediccion['gan_acum'] = tb_prediccion['ganancia'].cumsum()
        tb_prediccion = tb_prediccion.drop(columns=['ganancia'])
        
        # Acumulo las ganancias
        for icor, corte in enumerate(PARAM['train_final']['cortes']):
            if corte <= len(tb_prediccion):
                mganancias[vapo, icor] = tb_prediccion.iloc[corte - 1]['gan_acum']
        
        # Grabo las probabilidades del modelo
        tb_prediccion.to_csv(
            "prediccion.txt",
            sep='\t',
            index=False,
            mode='a',
            header=not os.path.exists("prediccion.txt") or os.path.getsize("prediccion.txt") == 0
        )
        
        del tb_prediccion
        gc.collect()

print(f"Predicciones completadas: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# RESULTADOS Y ARCHIVOS DE KAGGLE
# ============================================================================

print("\nMatriz de ganancias:")
print(mganancias)

# Creo directorio kaggle
os.makedirs("kaggle", exist_ok=True)

# Leo predicciones
tb_prediccion = pd.read_csv("prediccion.txt", sep='\t')

# Genero archivos de envío
envios = 11000

for vapo in range(PARAM['train_final']['APO']):
    tb_pred = tb_prediccion[tb_prediccion['meta_modelo'] == vapo + 1].copy()
    
    if len(tb_pred) > 0:
        tb_pred = tb_pred.sort_values('prob', ascending=False).reset_index(drop=True)
        tb_pred['Predicted'] = 0
        tb_pred.loc[:envios-1, 'Predicted'] = 1
        
        archivo_kaggle = f"./kaggle/KA{PARAM['experimento']}_{vapo+1}_{envios}.csv"
        tb_pred[['numero_de_cliente', 'Predicted']].to_csv(
            archivo_kaggle,
            index=False
        )
        
        del tb_pred
        gc.collect()

# Estadísticas finales
colmedias = mganancias.mean(axis=0)
mcorte_mejor = colmedias.max()
icorte_mejor = colmedias.argmax()
corte_mejor = PARAM['train_final']['cortes'][icorte_mejor]

print(f"\nMejor corte: {corte_mejor} con ganancia media: {mcorte_mejor:.2f}")

# Guardo estadísticas
tbl_medias = pd.DataFrame([colmedias], columns=[f"e{c}" for c in PARAM['train_final']['cortes']])
tbl_medias['experimento'] = PARAM['experimento']

exp_gral = "/content/buckets/b1/exp/apo-gral"
os.makedirs(exp_gral, exist_ok=True)

tbl_medias.to_csv(
    f"{exp_gral}/tb_experimentos.txt",
    sep='\t',
    index=False,
    mode='a',
    header=not os.path.exists(f"{exp_gral}/tb_experimentos.txt")
)

# Guardo tabla local
tbl_local = pd.DataFrame(
    mganancias,
    columns=[f"e{c}" for c in PARAM['train_final']['cortes']]
)
tbl_local.to_csv("tb_apo.txt", sep='\t', index=False)

# Genero el archivo de mejor envío
icerca = (tb_prediccion['gan_acum'] - mcorte_mejor).abs().idxmin()
vmodelo = tb_prediccion.loc[icerca, 'meta_modelo']
tb_pred_mejor = tb_prediccion[tb_prediccion['meta_modelo'] == vmodelo].copy()

icerca_mejor = (tb_pred_mejor['gan_acum'] - mcorte_mejor).abs().idxmin()
tb_pred_mejor['Predicted'] = 0
tb_pred_mejor.loc[:icerca_mejor, 'Predicted'] = 1

archivo_pseudo_kaggle = f"./kaggle/KA{PARAM['experimento']}_{icerca_mejor+1}.csv"
tb_pred_mejor[['numero_de_cliente', 'Predicted']].to_csv(
    archivo_pseudo_kaggle,
    index=False
)

print(f"\nArchivo final generado: {archivo_pseudo_kaggle}")
print(f"Número de envíos: {icerca_mejor + 1}")

print(f"\nFin: {time.strftime('%Y-%m-%d %H:%M:%S')}")
