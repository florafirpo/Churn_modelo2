#importar y armar flujo de funcion undersampling para balancear dataset
import pandas as pd
import numpy as np
import logging
from src.config import FILE_INPUT_DATA, FRACCION_SUBSAMPLEO, SEMILLA
from src.loader import cargar_datos
from src.preprocesamiento import undersampling


logger = logging.getLogger(__name__)

def lanzar_subsampleo(df:pd.DataFrame|np.ndarray, FRACCION_SUBSAMPLEO:float, SEMILLA:int) -> pd.DataFrame:
    """
    Función para lanzar el proceso de subsampleo en el dataset completo.
    df: DataFrame original
    FRACCION_SUBSAMPLEO: fracción de la clase mayoritaria a conservar
    SEMILLA: semilla para reproducibilidad
    Retorna el DataFrame balanceado.
    """
    ## 0. load datos
    df=cargar_datos(FILE_INPUT_DATA)
    print(df.head())


    logger.info(f"Inicio del proceso de subsampleo con fracción {FRACCION_SUBSAMPLEO} y semilla {SEMILLA}")
    df_balanceado = undersampling(df, FRACCION_SUBSAMPLEO, SEMILLA)
    logger.info("Finalización del proceso de subsampleo")
    #cuantos casos quedaron de cada clase
    logger.info(f"Total de registros después del subsampleo: {len(df_balanceado)}")
    logger.info(f"Total de y=1 : {(df_balanceado['clase_binaria']==1).sum()}")
    logger.info(f"Total de y=0 : {(df_balanceado['clase_binaria']==0).sum()}")

    #Guardo CSV del dataset balanceado con numero de fracción en el nombre y semilla
    output_path = f"data/processed/dataset_balanceado_fraccion_{FRACCION_SUBSAMPLEO}_semilla_{SEMILLA}.csv"
    df_balanceado.to_csv(output_path, index=False)
    return df_balanceado