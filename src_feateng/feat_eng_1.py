import numpy as np
import pandas as pd
import logging
import json
from src.config import *
from src.configuracion_inicial import creacion_df_small
from src.constr_lista_cols import contruccion_cols
from src.feature_engineering import feature_engineering_lag,feature_engineering_delta,feature_engineering_max_min,feature_engineering_ratio,feature_engineering_linreg,feature_engineering_drop_cols,feature_engineering_rank, feature_engineering_percentil,feature_engineering_drop_meses, suma_de_prod_servs, suma_ganancias_gastos, ratios_ganancia_gastos, cols_conteo_servicios_productos, cols_beneficios_presion_economica
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## Carga de variables

logger=logging.getLogger(__name__)

def lanzar_feat_eng(fecha:str ,n_fe:int , proceso_ppal:str):
    numero=n_fe
    #"""----------------------------------------------------------------------------------------------"""
    name=f"FEAT_ENG_{numero}_{proceso_ppal}_VENTANA_{VENTANA}"
    logger.info(f"PROCESO PRINCIPAL ---> {proceso_ppal}")
    logger.info(f"Comienzo del experimento : {name}")
    df_chiquito=creacion_df_small()

    # SELECCION DE COLUMNAS
    columnas=contruccion_cols(df_chiquito)
    cols_lag_delta_max_min_regl=columnas[0]
    cols_ratios=columnas[1]

        #DROPEO INICIAL DE MESES

    meses_a_dropear=[201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908, 201909, 201910, 201911, 201912, 202006]
    feature_engineering_drop_meses(meses_a_dropear,"df_completo","df_completo")
    

    # CORRECCION DE VARIABLES POR MES POR MEDIA
    # df_completo_chiquito=creacion_df_small("df_completo")
    # variable_meses_dict={"mrentabilidad":"(202106,202105)"}
    # feature_engineering_correccion_variables_por_mes_por_media(df_completo_chiquito,variable_meses_dict)

    # SERVICIOS Y PRODUCTOS
    df_completo_chiquito=creacion_df_small("df_completo")
    dict_prod_serv=cols_conteo_servicios_productos(df_completo_chiquito)
    for p_s, cols in dict_prod_serv.items():
        suma_de_prod_servs(df_completo_chiquito,cols,p_s)
    
    # GANANCIAS Y GASTOS
    ganancias_gastos=cols_beneficios_presion_economica(df_completo_chiquito)
    suma_ganancias_gastos(df_completo_chiquito,ganancias_gastos["ganancias"] , ganancias_gastos["gastos"])
    ratios_ganancia_gastos(df_completo_chiquito)


    # PERCENTIL
    # df_completo_chiquito=creacion_df_small("df_completo") # Para agregar las columnas de las corregidas
    cols_percentil,_,_=contruccion_cols(df_completo_chiquito)
    feature_engineering_percentil(df_completo_chiquito ,cols_percentil,bins=20)

    # RATIOS
    df_completo_chiquito=creacion_df_small("df_completo")
    _,_,cols_ratios = contruccion_cols(df_completo_chiquito)
    feature_engineering_ratio(df_completo_chiquito,cols_ratios)
 
     
    df_completo_chiquito=creacion_df_small("df_completo")
    _,  cols_lag_delta_max_min_regl  ,   _ = contruccion_cols(df_completo_chiquito)
    feature_engineering_lag(df_completo_chiquito,cols_lag_delta_max_min_regl,VENTANA)
    feature_engineering_delta(df_completo_chiquito,cols_lag_delta_max_min_regl,VENTANA)
    feature_engineering_linreg(df_completo_chiquito , cols_lag_delta_max_min_regl,VENTANA)
    feature_engineering_max_min(df_completo_chiquito,cols_lag_delta_max_min_regl ,VENTANA)
    

    # FEATURE ENGINEERING
    feature_engineering_lag(df_chiquito,cols_lag_delta_max_min_regl,VENTANA)
    feature_engineering_delta(df_chiquito,cols_lag_delta_max_min_regl,VENTANA)
    feature_engineering_ratio(df_chiquito,cols_ratios)
    feature_engineering_linreg(df_chiquito , cols_lag_delta_max_min_regl,VENTANA)
    feature_engineering_max_min(df_chiquito,cols_lag_delta_max_min_regl ,VENTANA)
    

    #DROPEO DE COLULNAS
    # cols_a_dropear=["mcuentas_saldo"]
    feature_engineering_drop_cols(df_chiquito,columnas=None)

    logger.info("================ FIN DEL PROCESO DE FEAT ENG =============================")





    

    
