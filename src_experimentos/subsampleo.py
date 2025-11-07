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

def undersampling_por_cliente(df: pd.DataFrame, undersampling_rate: float, semilla: int) -> pd.DataFrame:
    """
    Undersampling a nivel de CLIENTE (mantiene o elimina todos los registros de un cliente).
    USAR CON PRECAUCIÓN: Puede eliminar muchos datos.
    """
    logger.info("Comienzo del undersampling POR CLIENTE")
    np.random.seed(semilla)
    
    # Obtener clientes únicos por clase
    # Un cliente se considera de la clase que tiene en la MAYORÍA de sus registros
    cliente_clase = df.groupby('numero_de_cliente')['clase_ternaria'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    )
    clientes_minoritaria = df.loc[df["clase_ternaria"] != "Continua", "numero_de_cliente"].unique()
    clientes_mayoritaria = df.loc[df["clase_ternaria"] == "Continua", "numero_de_cliente"].unique()

    logger.info(f"Clientes minoritarios: {len(clientes_minoritaria):,}")
    logger.info(f"Clientes mayoritarios: {len(clientes_mayoritaria):,}")
    
    # Samplear clientes
    n_sample = int(len(clientes_mayoritaria) * undersampling_rate)
    clientes_mayoritaria_sample = np.random.choice(
        clientes_mayoritaria, 
        n_sample, 
        replace=False
    )
    
    # Unir clientes seleccionados
    clientes_finales = np.concatenate([clientes_minoritaria, clientes_mayoritaria_sample])
    
    # Filtrar DataFrame
    df_undersampled = df[df["numero_de_cliente"].isin(clientes_finales)].copy()
    df_undersampled = df_undersampled.sample(frac=1, random_state=semilla).reset_index(drop=True)
    
    logger.info(f"Shape original: {df.shape}")
    logger.info(f"Shape undersampled: {df_undersampled.shape}")
    logger.info(f"Registros minoritaria: {(df_undersampled['clase_ternaria'] != 'Continua').sum():,}")
    logger.info(f"Registros mayoritaria: {(df_undersampled['clase_ternaria'] == 'Continua').sum():,}")
    
    return df_undersampled

#   ESTO N ESTÁ BIEN PORQUE REQUERIRIA FILTRAR LOS DE LOS ULTIMOSMESES. tENDRIA que ser sólo undersampling entre los meses de 2019 y 202103
 