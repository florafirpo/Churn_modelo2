"""
Feature Engineering Functions - Optimizado con Checkpoints
Diseñado para datasets grandes con capacidad de recuperación ante desconexiones
"""

import duckdb
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Optional

from src.config import PATH_DATA_BASE_DB, PATH_OUTPUT_DATA

logger = logging.getLogger(__name__)


# ============================================
# SISTEMA DE CHECKPOINTS
# ============================================

class CheckpointManager:
    """Gestiona checkpoints para recuperación ante fallas"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.checkpoint_dir = self.base_path / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        
    def _load_metadata(self) -> Dict:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self, metadata: Dict):
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def existe_checkpoint(self, nombre_etapa: str) -> bool:
        metadata = self._load_metadata()
        return nombre_etapa in metadata and metadata[nombre_etapa].get('completado', False)
    
    def guardar_checkpoint(self, nombre_etapa: str, info_adicional: Dict = None):
        try:
            conn = duckdb.connect(PATH_DATA_BASE_DB)
            count = conn.execute("SELECT COUNT(*) FROM df_completo").fetchone()[0]
            
            checkpoint_path = self.checkpoint_dir / f"{nombre_etapa}.parquet"
            conn.execute(f"COPY df_completo TO '{checkpoint_path}' (FORMAT PARQUET)")
            conn.close()
            
            metadata = self._load_metadata()
            metadata[nombre_etapa] = {
                'completado': True,
                'timestamp': datetime.now().isoformat(),
                'registros': count,
                'archivo': str(checkpoint_path),
                **(info_adicional or {})
            }
            self._save_metadata(metadata)
            
            logger.info(f"💾 Checkpoint guardado: {nombre_etapa} ({count:,} registros)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error guardando checkpoint {nombre_etapa}: {e}")
            return False
    
    def cargar_checkpoint(self, nombre_etapa: str) -> bool:
        try:
            metadata = self._load_metadata()
            
            if nombre_etapa not in metadata or not metadata[nombre_etapa].get('completado'):
                return False
            
            checkpoint_path = Path(metadata[nombre_etapa]['archivo'])
            
            if not checkpoint_path.exists():
                logger.warning(f"⚠️ Archivo de checkpoint no encontrado: {checkpoint_path}")
                return False
            
            conn = duckdb.connect(PATH_DATA_BASE_DB)
            conn.execute("DROP TABLE IF EXISTS df_completo")
            conn.execute(f"CREATE TABLE df_completo AS SELECT * FROM read_parquet('{checkpoint_path}')")
            conn.close()
            
            registros = metadata[nombre_etapa]['registros']
            logger.info(f"📂 Checkpoint cargado: {nombre_etapa} ({registros:,} registros)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando checkpoint {nombre_etapa}: {e}")
            return False
    
    def listar_checkpoints(self) -> Dict:
        return self._load_metadata()


# ============================================
# UTILIDADES
# ============================================

def _ejecutar_sql(sql: str, descripcion: str):
    """Ejecuta SQL con logs de tiempo"""
    inicio = datetime.now()
    logger.info(f"⏳ {descripcion}...")
    
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        conn.execute(sql)
        conn.close()
        
        duracion = (datetime.now() - inicio).total_seconds()
        logger.info(f"✅ {descripcion} completado ({duracion:.2f}s)")
        return True
    except Exception as e:
        logger.error(f"❌ Error en {descripcion}: {e}")
        raise


def _validar_columnas(columnas: List[str]) -> List[str]:
    """Valida qué columnas existen en la tabla"""
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    columnas_existentes = conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'df_completo'"
    ).df()['column_name'].tolist()
    conn.close()
    
    columnas_validas = [col for col in columnas if col in columnas_existentes]
    columnas_faltantes = set(columnas) - set(columnas_validas)
    
    if columnas_faltantes:
        logger.warning(f"⚠️ Columnas no encontradas: {list(columnas_faltantes)[:5]}...")
    
    return columnas_validas


# ============================================
# FUNCIONES DE FEATURE ENGINEERING
# ============================================

def feature_engineering_drop_meses(meses: List[int], tabla_input: str, tabla_output: str, 
                                   checkpoint_mgr: Optional[CheckpointManager] = None):
    """Elimina meses específicos"""
    etapa = "01_drop_meses"
    
    if checkpoint_mgr and checkpoint_mgr.existe_checkpoint(etapa):
        logger.info(f"⏩ Saltando: {etapa}")
        checkpoint_mgr.cargar_checkpoint(etapa)
        return
    
    meses_str = ", ".join(map(str, meses))
    sql = f"""
        CREATE OR REPLACE TABLE {tabla_output} AS
        SELECT * FROM {tabla_input}
        WHERE foto_mes NOT IN ({meses_str})
    """
    
    _ejecutar_sql(sql, f"Eliminando meses {meses}")
    
    if checkpoint_mgr:
        checkpoint_mgr.guardar_checkpoint(etapa, {'meses_eliminados': meses})


def suma_de_prod_servs(df: pd.DataFrame, columnas: List[str], nombre_feature: str,
                       checkpoint_mgr: Optional[CheckpointManager] = None):
    """Suma productos/servicios y crea nueva columna"""
    etapa = f"02_suma_{nombre_feature}"
    
    if checkpoint_mgr and checkpoint_mgr.existe_checkpoint(etapa):
        logger.info(f"⏩ Saltando: {etapa}")
        return
    
    cols_validas = _validar_columnas(columnas)
    
    if not cols_validas:
        logger.warning(f"⚠️ No hay columnas válidas para {nombre_feature}")
        return
    
    # Filtrar solo columnas numéricas
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    cols_info = conn.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'df_completo'
    """).df()
    conn.close()
    
    cols_numericas = cols_info[
        cols_info['data_type'].str.contains('INT|DOUBLE|FLOAT|DECIMAL|NUMERIC', case=False, na=False)
    ]['column_name'].tolist()
    
    cols_validas_numericas = [col for col in cols_validas if col in cols_numericas]
    
    if not cols_validas_numericas:
        logger.warning(f"⚠️ No hay columnas numéricas válidas para {nombre_feature}")
        return
    
    # Usar COALESCE solo en la suma final, no en las columnas originales
    # Esto suma ignorando NULLs: si col1=NULL, col2=5, col3=3 → resultado=8
    sql = f"""
        CREATE OR REPLACE TABLE df_completo AS
        SELECT *,
            ({' + '.join([f'COALESCE({col}, 0)' for col in cols_validas_numericas])}) AS total_{nombre_feature}
        FROM df_completo
    """
    
    _ejecutar_sql(sql, f"Creando total_{nombre_feature}")
    
    if checkpoint_mgr:
        checkpoint_mgr.guardar_checkpoint(etapa)


def suma_ganancias_gastos(df: pd.DataFrame, cols_ganancias: List[str], cols_gastos: List[str],
                         checkpoint_mgr: Optional[CheckpointManager] = None):
    """Crea features de ganancias, gastos y ratios"""
    etapa = "03_ganancias_gastos"
    
    if checkpoint_mgr and checkpoint_mgr.existe_checkpoint(etapa):
        logger.info(f"⏩ Saltando: {etapa}")
        return
    
    ganancias_validas = _validar_columnas(cols_ganancias)
    gastos_validos = _validar_columnas(cols_gastos)
    
    # Filtrar solo columnas numéricas
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    cols_info = conn.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'df_completo'
    """).df()
    conn.close()
    
    cols_numericas = cols_info[
        cols_info['data_type'].str.contains('INT|DOUBLE|FLOAT|DECIMAL|NUMERIC', case=False, na=False)
    ]['column_name'].tolist()
    
    ganancias_numericas = [col for col in ganancias_validas if col in cols_numericas]
    gastos_numericos = [col for col in gastos_validos if col in cols_numericas]
    
    if not ganancias_numericas or not gastos_numericos:
        logger.warning("⚠️ No hay suficientes columnas numéricas para ganancias/gastos")
        return
    
    # COALESCE solo en la suma, ignorando NULLs
    sql = f"""
        CREATE OR REPLACE TABLE df_completo AS
        SELECT *,
            ({' + '.join([f'COALESCE({col}, 0)' for col in ganancias_numericas])}) AS total_ganancias,
            ({' + '.join([f'COALESCE({col}, 0)' for col in gastos_numericos])}) AS total_gastos,
            ({' + '.join([f'COALESCE({col}, 0)' for col in ganancias_numericas])}) - 
            ({' + '.join([f'COALESCE({col}, 0)' for col in gastos_numericos])}) AS ganancia_neta
        FROM df_completo
    """
    
    _ejecutar_sql(sql, "Creando features de ganancias/gastos")
    
    if checkpoint_mgr:
        checkpoint_mgr.guardar_checkpoint(etapa)


def ratios_ganancia_gastos(df: pd.DataFrame, checkpoint_mgr: Optional[CheckpointManager] = None):
    """Crea ratio ganancia/gasto"""
    etapa = "04_ratio_ganancia_gasto"
    
    if checkpoint_mgr and checkpoint_mgr.existe_checkpoint(etapa):
        logger.info(f"⏩ Saltando: {etapa}")
        return
    
    sql = """
        CREATE OR REPLACE TABLE df_completo AS
        SELECT *,
            CASE 
                WHEN total_gastos != 0 AND total_gastos IS NOT NULL
                THEN total_ganancias / total_gastos
                ELSE NULL
            END AS ratio_ganancia_gasto
        FROM df_completo
    """
    
    _ejecutar_sql(sql, "Creando ratio ganancia/gasto")
    
    if checkpoint_mgr:
        checkpoint_mgr.guardar_checkpoint(etapa)


def feature_engineering_percentil(df: pd.DataFrame, columnas: List[str], bins: int = 20,
                                  checkpoint_mgr: Optional[CheckpointManager] = None):
    """Crea features de percentiles por foto_mes"""
    etapa = "05_percentiles"
    
    if checkpoint_mgr and checkpoint_mgr.existe_checkpoint(etapa):
        logger.info(f"⏩ Saltando: {etapa}")
        return
    
    cols_validas = _validar_columnas(columnas)
    
    if not cols_validas:
        logger.warning("⚠️ No hay columnas válidas para percentiles")
        return
    
    # Limitar a 50 columnas para evitar queries muy largas
    cols_a_procesar = cols_validas[:50]
    
    ntile_cols = [
        f"NTILE({bins}) OVER (PARTITION BY foto_mes ORDER BY {col}) AS {col}_percentil"
        for col in cols_a_procesar
    ]
    
    sql = f"""
        CREATE OR REPLACE TABLE df_completo AS
        SELECT *,
            {', '.join(ntile_cols)}
        FROM df_completo
    """
    
    _ejecutar_sql(sql, f"Creando percentiles ({len(cols_a_procesar)} columnas)")
    
    if checkpoint_mgr:
        checkpoint_mgr.guardar_checkpoint(etapa, {'bins': bins, 'columnas': len(cols_a_procesar)})


def feature_engineering_ratio(df: pd.DataFrame, pares_columnas: List[List[str]],
                              checkpoint_mgr: Optional[CheckpointManager] = None):
    """Crea features de ratios entre pares de columnas"""
    etapa = "06_ratios"
    
    if checkpoint_mgr and checkpoint_mgr.existe_checkpoint(etapa):
        logger.info(f"⏩ Saltando: {etapa}")
        return
    
    if not pares_columnas:
        logger.info("ℹ️ No hay pares para crear ratios")
        return
    
    ratio_expressions = []
    for par in pares_columnas:
        if len(par) == 2:
            col1, col2 = par
            ratio_expressions.append(f"""
                CASE 
                    WHEN {col2} != 0 AND {col2} IS NOT NULL
                    THEN {col1} / {col2}
                    ELSE NULL
                END AS ratio_{col1}_{col2}
            """)
    
    if not ratio_expressions:
        return
    
    sql = f"""
        CREATE OR REPLACE TABLE df_completo AS
        SELECT *,
            {', '.join(ratio_expressions)}
        FROM df_completo
    """
    
    _ejecutar_sql(sql, f"Creando ratios ({len(ratio_expressions)} pares)")
    
    if checkpoint_mgr:
        checkpoint_mgr.guardar_checkpoint(etapa, {'pares': len(ratio_expressions)})


def feature_engineering_lag(df: pd.DataFrame, columnas: List[str], ventana: int = 3,
                            checkpoint_mgr: Optional[CheckpointManager] = None):
    """Crea features de LAGs"""
    etapa = f"07_lags_v{ventana}"
    
    if checkpoint_mgr and checkpoint_mgr.existe_checkpoint(etapa):
        logger.info(f"⏩ Saltando: {etapa}")
        return
    
    cols_validas = _validar_columnas(columnas)
    
    if not cols_validas:
        logger.warning("⚠️ No hay columnas válidas para lags")
        return
    
    lag_expressions = []
    for col in cols_validas:
        for i in range(1, ventana + 1):
            lag_expressions.append(
                f"LAG({col}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {col}_lag_{i}"
            )
    
    sql = f"""
        CREATE OR REPLACE TABLE df_completo AS
        SELECT *,
            {', '.join(lag_expressions)}
        FROM df_completo
    """
    
    _ejecutar_sql(sql, f"Creando lags ({len(cols_validas)} cols x {ventana} lags)")
    
    if checkpoint_mgr:
        checkpoint_mgr.guardar_checkpoint(etapa, {'ventana': ventana, 'columnas': len(cols_validas)})


def feature_engineering_delta(df: pd.DataFrame, columnas: List[str], ventana: int = 3,
                              checkpoint_mgr: Optional[CheckpointManager] = None):
    """Crea features de DELTAs (diferencias con lags)"""
    etapa = f"08_deltas_v{ventana}"
    
    if checkpoint_mgr and checkpoint_mgr.existe_checkpoint(etapa):
        logger.info(f"⏩ Saltando: {etapa}")
        return
    
    cols_validas = _validar_columnas(columnas)
    
    if not cols_validas:
        logger.warning("⚠️ No hay columnas válidas para deltas")
        return
    
    delta_expressions = []
    for col in cols_validas:
        for i in range(1, ventana + 1):
            # Verificar si existe el lag
            if f"{col}_lag_{i}" in _validar_columnas([f"{col}_lag_{i}"]):
                delta_expressions.append(
                    f"{col} - {col}_lag_{i} AS {col}_delta_{i}"
                )
    
    if not delta_expressions:
        logger.warning("⚠️ No se encontraron lags previos para crear deltas")
        return
    
    sql = f"""
        CREATE OR REPLACE TABLE df_completo AS
        SELECT *,
            {', '.join(delta_expressions)}
        FROM df_completo
    """
    
    _ejecutar_sql(sql, f"Creando deltas ({len(delta_expressions)} deltas)")
    
    if checkpoint_mgr:
        checkpoint_mgr.guardar_checkpoint(etapa, {'ventana': ventana})


def feature_engineering_linreg(df: pd.DataFrame, columnas: List[str], ventana: int = 3,
                               checkpoint_mgr: Optional[CheckpointManager] = None):
    """Crea features de SLOPE (regresión lineal)"""
    etapa = f"09_slopes_v{ventana}"
    
    if checkpoint_mgr and checkpoint_mgr.existe_checkpoint(etapa):
        logger.info(f"⏩ Saltando: {etapa}")
        return
    
    cols_validas = _validar_columnas(columnas)
    
    if not cols_validas:
        logger.warning("⚠️ No hay columnas válidas para slopes")
        return
    
    slope_expressions = [
        f"REGR_SLOPE({col}, foto_mes) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN {ventana} PRECEDING AND CURRENT ROW) AS {col}_slope"
        for col in cols_validas
    ]
    
    sql = f"""
        CREATE OR REPLACE TABLE df_completo AS
        SELECT *,
            {', '.join(slope_expressions)}
        FROM df_completo
    """
    
    _ejecutar_sql(sql, f"Creando slopes ({len(cols_validas)} columnas)")
    
    if checkpoint_mgr:
        checkpoint_mgr.guardar_checkpoint(etapa, {'ventana': ventana, 'columnas': len(cols_validas)})


def feature_engineering_max_min(df: pd.DataFrame, columnas: List[str], ventana: int = 3,
                                checkpoint_mgr: Optional[CheckpointManager] = None):
    """Crea features de MAX y MIN en ventanas móviles"""
    etapa = f"10_max_min_v{ventana}"
    
    if checkpoint_mgr and checkpoint_mgr.existe_checkpoint(etapa):
        logger.info(f"⏩ Saltando: {etapa}")
        return
    
    cols_validas = _validar_columnas(columnas)
    
    if not cols_validas:
        logger.warning("⚠️ No hay columnas válidas para max/min")
        return
    
    max_min_expressions = []
    for col in cols_validas:
        max_min_expressions.extend([
            f"MAX({col}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN {ventana} PRECEDING AND CURRENT ROW) AS {col}_max_{ventana}",
            f"MIN({col}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN {ventana} PRECEDING AND CURRENT ROW) AS {col}_min_{ventana}"
        ])
    
    sql = f"""
        CREATE OR REPLACE TABLE df_completo AS
        SELECT *,
            {', '.join(max_min_expressions)}
        FROM df_completo
    """
    
    _ejecutar_sql(sql, f"Creando max/min ({len(cols_validas)} columnas)")
    
    if checkpoint_mgr:
        checkpoint_mgr.guardar_checkpoint(etapa, {'ventana': ventana, 'columnas': len(cols_validas)})


def feature_engineering_drop_cols(df: pd.DataFrame, columnas: Optional[List[str]] = None,
                                  checkpoint_mgr: Optional[CheckpointManager] = None):
    """Elimina columnas especificadas"""
    etapa = "11_drop_cols"
    
    if checkpoint_mgr and checkpoint_mgr.existe_checkpoint(etapa):
        logger.info(f"⏩ Saltando: {etapa}")
        return
    
    if not columnas:
        logger.info("ℹ️ No hay columnas para eliminar")
        if checkpoint_mgr:
            checkpoint_mgr.guardar_checkpoint(etapa)
        return
    
    cols_validas = _validar_columnas(columnas)
    
    if not cols_validas:
        logger.warning("⚠️ Ninguna de las columnas a eliminar existe")
        if checkpoint_mgr:
            checkpoint_mgr.guardar_checkpoint(etapa)
        return
    
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    todas_las_columnas = conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'df_completo'"
    ).df()['column_name'].tolist()
    conn.close()
    
    columnas_mantener = [col for col in todas_las_columnas if col not in cols_validas]
    
    sql = f"""
        CREATE OR REPLACE TABLE df_completo AS
        SELECT {', '.join(columnas_mantener)}
        FROM df_completo
    """
    
    _ejecutar_sql(sql, f"Eliminando {len(cols_validas)} columnas")
    
    if checkpoint_mgr:
        checkpoint_mgr.guardar_checkpoint(etapa, {'columnas_eliminadas': cols_validas})