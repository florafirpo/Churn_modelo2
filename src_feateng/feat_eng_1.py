"""
Pipeline de Feature Engineering - Optimizado con Sistema de Checkpoints
Diseñado para datasets grandes con recuperación automática ante desconexiones
"""

import logging
from datetime import datetime

from src.config import VENTANA, PATH_OUTPUT_DATA
from src.configuracion_inicial import creacion_df_small
from src.constr_lista_cols import (
    contruccion_cols, 
    cols_conteo_servicios_productos, 
    cols_beneficios_presion_economica
)
from src.feat_eng_optimized import (
    CheckpointManager,
    feature_engineering_drop_meses,
    suma_de_prod_servs,
    suma_ganancias_gastos,
    ratios_ganancia_gastos,
    feature_engineering_percentil,
    feature_engineering_ratio,
    feature_engineering_lag,
    feature_engineering_delta,
    feature_engineering_linreg,
    feature_engineering_max_min,
    feature_engineering_drop_cols
)

logger = logging.getLogger(__name__)


def lanzar_feat_eng(fecha: str, n_fe: int, proceso_ppal: str, usar_checkpoints: bool = False):
    """
    Pipeline principal de Feature Engineering con sistema de checkpoints OPCIONAL.
    
    Args:
        fecha: Fecha del experimento
        n_fe: Número de feature engineering
        proceso_ppal: Nombre del proceso principal
        usar_checkpoints: Si True, guarda checkpoints (más lento pero seguro). 
                         Si False, no guarda checkpoints (más rápido).
    """
    
    # ============================================
    # INICIALIZACIÓN
    # ============================================
    
    name = f"FEAT_ENG_{n_fe}_{proceso_ppal}_VENTANA_{VENTANA}"
    logger.info("=" * 80)
    logger.info(f"PROCESO PRINCIPAL: {proceso_ppal}")
    logger.info(f"EXPERIMENTO: {name}")
    logger.info(f"Fecha: {fecha}")
    logger.info(f"Checkpoints: {'ACTIVADOS' if usar_checkpoints else 'DESACTIVADOS (modo rápido)'}")
    logger.info("=" * 80)
    
    inicio_total = datetime.now()
    
    # Inicializar gestor de checkpoints solo si está activado
    checkpoint_mgr = CheckpointManager(PATH_OUTPUT_DATA) if usar_checkpoints else None
    
    if checkpoint_mgr:
        checkpoints_previos = checkpoint_mgr.listar_checkpoints()
        if checkpoints_previos:
            logger.info(f"📋 Checkpoints encontrados: {len(checkpoints_previos)}")
    else:
        logger.info("⚡ Modo rápido: Sin checkpoints")
    
    # ============================================
    # ANÁLISIS INICIAL DE COLUMNAS - UNA SOLA VEZ
    # ============================================
    
    logger.info("\n🔍 Analizando estructura de datos (una sola vez)...")
    df_sample = creacion_df_small()
    
    # Extraer TODAS las columnas necesarias en una sola pasada
    columnas = contruccion_cols(df_sample)
    cols_lag_delta_max_min_regl = columnas[0]
    cols_ratios = columnas[1]
    
    dict_prod_serv = cols_conteo_servicios_productos(df_sample)
    ganancias_gastos = cols_beneficios_presion_economica(df_sample)
    
    # Extraer columnas para percentiles desde el principio
    cols_percentil = cols_lag_delta_max_min_regl.copy()  # Usar las mismas que lag/delta
    
    logger.info(f"✓ Columnas lag/delta: {len(cols_lag_delta_max_min_regl)}")
    logger.info(f"✓ Pares de ratios: {len(cols_ratios)}")
    logger.info(f"✓ Grupos prod/serv: {len(dict_prod_serv)}")
    logger.info(f"✓ Columnas percentiles: {len(cols_percentil)}")
    
    # ============================================
    # ETAPA 1: DROPEO INICIAL DE MESES
    # ============================================
    
    logger.info("\n" + "=" * 80)
    logger.info("ETAPA 1: Eliminando meses antiguos")
    logger.info("=" * 80)
    
    meses_a_dropear = [201901, 201902, 201903, 201904, 201905, 
                       201906, 201907, 201908, 201909]
    
    feature_engineering_drop_meses(
        meses_a_dropear, 
        "df_completo", 
        "df_completo",
        checkpoint_mgr
    )
    
    # ============================================
    # ETAPA 2: PRODUCTOS Y SERVICIOS
    # ============================================
    
    logger.info("\n" + "=" * 80)
    logger.info("ETAPA 2: Creando features de productos y servicios")
    logger.info("=" * 80)
    
    # Usar las columnas ya extraídas, no volver a leer
    for nombre_grupo, cols in dict_prod_serv.items():
        suma_de_prod_servs(
            df_sample,  # Pasamos el sample pero solo se usan las columnas
            cols, 
            nombre_grupo,
            checkpoint_mgr
        )
    
    # ============================================
    # ETAPA 3: GANANCIAS Y GASTOS
    # ============================================
    
    logger.info("\n" + "=" * 80)
    logger.info("ETAPA 3: Creando features de ganancias y gastos")
    logger.info("=" * 80)
    
    suma_ganancias_gastos(
        df_sample,
        ganancias_gastos["ganancias"],
        ganancias_gastos["gastos"],
        checkpoint_mgr
    )
    
    ratios_ganancia_gastos(df_sample, checkpoint_mgr)
    
    # ============================================
    # ETAPA 4: PERCENTILES
    # ============================================
    
    logger.info("\n" + "=" * 80)
    logger.info("ETAPA 4: Creando percentiles")
    logger.info("=" * 80)
    
    feature_engineering_percentil(
        df_sample,
        cols_percentil,  # Usar columnas ya extraídas
        bins=20,
        checkpoint_mgr=checkpoint_mgr
    )
    
    # ============================================
    # ETAPA 5: RATIOS
    # ============================================
    
    logger.info("\n" + "=" * 80)
    logger.info("ETAPA 5: Creando ratios")
    logger.info("=" * 80)
    
    feature_engineering_ratio(
        df_sample,
        cols_ratios,  # Usar columnas ya extraídas
        checkpoint_mgr
    )
    
    # ============================================
    # ETAPA 6: FEATURES TEMPORALES (LAG, DELTA, SLOPE, MAX/MIN)
    # ============================================
    
    logger.info("\n" + "=" * 80)
    logger.info("ETAPA 6: Creando features temporales")
    logger.info("=" * 80)
    logger.info(f"📊 Procesando {len(cols_lag_delta_max_min_regl)} columnas con ventana={VENTANA}")
    
    # LAGs
    feature_engineering_lag(
        df_sample,
        cols_lag_delta_max_min_regl,  # Usar columnas ya extraídas
        VENTANA,
        checkpoint_mgr
    )
    
    # DELTAs (requiere que los lags ya existan)
    feature_engineering_delta(
        df_sample,
        cols_lag_delta_max_min_regl,
        VENTANA,
        checkpoint_mgr
    )
    
    # SLOPE (regresión lineal)
    feature_engineering_linreg(
        df_sample,
        cols_lag_delta_max_min_regl,
        VENTANA,
        checkpoint_mgr
    )
    
    # MAX y MIN
    feature_engineering_max_min(
        df_sample,
        cols_lag_delta_max_min_regl,
        VENTANA,
        checkpoint_mgr
    )
    
    # ============================================
    # ETAPA 7: LIMPIEZA FINAL (OPCIONAL)
    # ============================================
    
    logger.info("\n" + "=" * 80)
    logger.info("ETAPA 7: Limpieza final")
    logger.info("=" * 80)
    
    cols_a_dropear = None
    
    feature_engineering_drop_cols(
        df_sample,
        columnas=cols_a_dropear,
        checkpoint_mgr=checkpoint_mgr
    )
    
    # ============================================
    # FINALIZACIÓN
    # ============================================
    
    duracion_total = (datetime.now() - inicio_total).total_seconds()
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ FEATURE ENGINEERING COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"Tiempo total: {duracion_total:.2f} segundos ({duracion_total/60:.2f} minutos)")
    logger.info(f"Experimento: {name}")
    
    if checkpoint_mgr:
        checkpoints_finales = checkpoint_mgr.listar_checkpoints()
        logger.info(f"\n📋 Total de etapas completadas: {len(checkpoints_finales)}")
    
    logger.info("=" * 80)