import duckdb
import pandas as pd
import logging
import yaml
from src.config import PATH_CONFIG, PATH_DATA_BASE_DB

# Configurar Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cargar_config():
    with open(PATH_CONFIG, "r") as f:
        return yaml.safe_load(f)

def obtener_columnas_numericas(conn, tabla):
    """Obtiene las columnas numéricas excluyendo las de control"""
    df_cols = conn.execute(f"DESCRIBE {tabla}").df()
    # Filtramos columnas que no sean numéricas o que sean IDs/Targets
    cfg = cargar_config()
    cols_no_lag = cfg['configuracion_general']['COLS_NO_LAG']
    
    cols = df_cols[
        (df_cols['column_name'].isin(cols_no_lag) == False) & 
        (df_cols['column_type'].str.contains('INT|DOUBLE|FLOAT|DECIMAL'))
    ]['column_name'].tolist()
    return cols

def generar_sql_sumas():
    """Genera el SQL para sumar tenencia de productos"""
    # Diccionario simplificado de productos
    grupos = {
        "tarjetas": ["ctarjeta_visa", "ctarjeta_master", "ctarjeta_debito"],
        "cuentas": ["ccuenta_corriente", "ccaja_ahorro", "cplazo_fijo", "cinversion1", "cinversion2"],
        "prestamos": ["cprestamos_personales", "cprestamos_prendarios", "cprestamos_hipotecarios"],
        "seguros": ["cseguro_vida", "cseguro_auto", "cseguro_vivienda", "cseguro_accidentes_personales"]
    }
    
    sql_parts = []
    for nombre_grupo, columnas in grupos.items():
        # Suma simple: si tiene el producto (valor > 0) suma 1, sino 0
        suma_logica = " + ".join([f"(CASE WHEN {col} > 0 THEN 1 ELSE 0 END)" for col in columnas])
        sql_parts.append(f"({suma_logica}) as cant_{nombre_grupo}")
    
    return ", ".join(sql_parts)

def lanzar_feat_eng():
    cfg = cargar_config()
    mes_corte = cfg['configuracion_general']['MES_INICIO_FILTRO']
    lag_max = cfg['configuracion_feat_eng']['LAG_ORDER']
    
    logger.info(f"🚀 Iniciando Feature Engineering Optimizado. DB: {PATH_DATA_BASE_DB}")
    
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    
    # ----------------------------------------------------------------
    # PASO 1: Limpieza inicial (Filtrar meses viejos)
    # ----------------------------------------------------------------
    logger.info(f"1. Eliminando registros anteriores a {mes_corte}...")
    # Es más rápido recrear la tabla filtrada que hacer un DELETE masivo
    conn.execute(f"""
        CREATE OR REPLACE TABLE df_base AS 
        SELECT * FROM df_completo 
        WHERE foto_mes > {mes_corte}
    """)
    
    count_orig = conn.execute("SELECT count(*) FROM df_completo").fetchone()[0]
    count_new = conn.execute("SELECT count(*) FROM df_base").fetchone()[0]
    logger.info(f"   Registros reducidos de {count_orig:,} a {count_new:,}")

    # ----------------------------------------------------------------
    # PASO 2: Construcción Dinámica de la Query Masiva
    # ----------------------------------------------------------------
    logger.info("2. Construyendo Query de Lags, Deltas y Sumas...")
    
    cols_a_laggear = obtener_columnas_numericas(conn, "df_base")
    logger.info(f"   Se generarán lags para {len(cols_a_laggear)} columnas.")

    select_clause = ["*"] # Traer todas las columnas originales
    
    # A. Agregar Sumas de productos
    if cfg['configuracion_feat_eng']['HACER_SUMAS']:
        select_clause.append(generar_sql_sumas())

    # B. Agregar Lags y Deltas
    for col in cols_a_laggear:
        for i in range(1, lag_max + 1):
            # Lags: valor en mes anterior
            lag_col = f"LAG({col}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)"
            select_clause.append(f"{lag_col} AS {col}_lag{i}")
            
            # Deltas: valor actual - lag
            # DuckDB maneja NULLs automáticamente (resultará en NULL)
            delta_sql = f"{col} - {lag_col}"
            select_clause.append(f"({delta_sql}) AS {col}_delta{i}")

    # Unimos todo el SELECT
    full_select = ", ".join(select_clause)
    
    # ----------------------------------------------------------------
    # PASO 3: Ejecución y Materialización
    # ----------------------------------------------------------------
    logger.info("3. Ejecutando transformaciones masivas (esto puede tardar unos minutos)...")
    
    # Creamos la tabla final directamente. 
    # DuckDB optimiza esto ejecutando en streaming sin cargar todo en RAM.
    query_final = f"""
        CREATE OR REPLACE TABLE df_fe AS
        SELECT 
            {full_select}
        FROM df_base
        ORDER BY numero_de_cliente, foto_mes
    """
    
    try:
        conn.execute(query_final)
        logger.info("   ✅ Tabla 'df_fe' creada exitosamente.")
        
        # Limpieza de tablas intermedias para ahorrar espacio si es necesario
        conn.execute("DROP TABLE IF EXISTS df_base")
        
        # Verificación rápida
        cols_finales = len(conn.execute("DESCRIBE df_fe").fetchall())
        logger.info(f"   📊 Dimensiones finales: {count_new:,} filas, {cols_finales} columnas.")
        
    except Exception as e:
        logger.error(f"❌ Error ejecutando SQL: {e}")
        raise
    finally:
        conn.close()

    logger.info("🏁 Feature Engineering finalizado.")

if __name__ == "__main__":
    lanzar_feat_eng()