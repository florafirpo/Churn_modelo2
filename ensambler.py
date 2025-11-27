"""
Ensamblador de Predicciones - Combina múltiples modelos
Diferentes estrategias de ensamble para mejorar predicciones

Autor: Data Scientist Junior
Fecha: 2025-11-26
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# ESTRATEGIAS DE ENSAMBLE
# ============================================================================

def ensamble_promedio_simple(predicciones_list, pesos=None):
    """
    Ensamble por PROMEDIO SIMPLE (o ponderado)
    
    Args:
        predicciones_list: Lista de DataFrames con columnas ['numero_de_cliente', 'prob']
        pesos: Lista de pesos para cada modelo (None = pesos iguales)
    
    Returns:
        DataFrame con predicciones ensambladas
    """
    logger.info("Estrategia: PROMEDIO SIMPLE")
    
    n_modelos = len(predicciones_list)
    
    # Validar que todos tengan los mismos clientes
    clientes_base = set(predicciones_list[0]['numero_de_cliente'])
    for i, pred in enumerate(predicciones_list[1:], 1):
        clientes_actual = set(pred['numero_de_cliente'])
        if clientes_base != clientes_actual:
            logger.warning(f"⚠️ Modelo {i+1} tiene clientes diferentes!")
    
    # Si no hay pesos, usar pesos iguales
    if pesos is None:
        pesos = [1.0] * n_modelos
        logger.info(f"  Usando pesos iguales: {pesos}")
    else:
        # Normalizar pesos para que sumen 1
        suma_pesos = sum(pesos)
        pesos = [p / suma_pesos for p in pesos]
        logger.info(f"  Pesos normalizados: {[f'{p:.3f}' for p in pesos]}")
    
    # Crear DataFrame base con el primer modelo
    resultado = predicciones_list[0][['numero_de_cliente']].copy()
    resultado['prob'] = predicciones_list[0]['prob'] * pesos[0]
    
    # Sumar las probabilidades de los demás modelos
    for i, (pred, peso) in enumerate(zip(predicciones_list[1:], pesos[1:]), 1):
        # Merge para asegurar orden correcto
        pred_temp = pred[['numero_de_cliente', 'prob']].copy()
        pred_temp = pred_temp.rename(columns={'prob': f'prob_{i}'})
        resultado = resultado.merge(pred_temp, on='numero_de_cliente', how='inner')
        resultado['prob'] += resultado[f'prob_{i}'] * peso
        resultado = resultado.drop(columns=[f'prob_{i}'])
    
    # Ordenar por probabilidad descendente
    resultado = resultado.sort_values('prob', ascending=False).reset_index(drop=True)
    
    logger.info(f"  ✓ Ensamble de {n_modelos} modelos completado")
    logger.info(f"  ✓ Total clientes: {len(resultado):,}")
    
    return resultado


def ensamble_rank_average(predicciones_list, pesos=None):
    """
    Ensamble por PROMEDIO DE RANKINGS
    Más robusto que el promedio de probabilidades
    
    Args:
        predicciones_list: Lista de DataFrames con columnas ['numero_de_cliente', 'prob']
        pesos: Lista de pesos para cada modelo (None = pesos iguales)
    
    Returns:
        DataFrame con predicciones ensambladas
    """
    logger.info("Estrategia: PROMEDIO DE RANKINGS")
    
    n_modelos = len(predicciones_list)
    
    # Si no hay pesos, usar pesos iguales
    if pesos is None:
        pesos = [1.0] * n_modelos
    else:
        suma_pesos = sum(pesos)
        pesos = [p / suma_pesos for p in pesos]
    
    logger.info(f"  Pesos: {[f'{p:.3f}' for p in pesos]}")
    
    # Crear ranking para cada modelo
    resultado = predicciones_list[0][['numero_de_cliente']].copy()
    resultado['rank_score'] = 0.0
    
    for i, (pred, peso) in enumerate(zip(predicciones_list, pesos)):
        # Calcular ranking (1 = mejor, N = peor)
        pred_rank = pred.copy()
        pred_rank = pred_rank.sort_values('prob', ascending=False).reset_index(drop=True)
        pred_rank['rank'] = range(1, len(pred_rank) + 1)
        
        # Normalizar rank entre 0 y 1 (1 = mejor)
        pred_rank['rank_norm'] = 1 - (pred_rank['rank'] - 1) / (len(pred_rank) - 1)
        
        # Merge y acumular
        pred_rank = pred_rank[['numero_de_cliente', 'rank_norm']]
        pred_rank = pred_rank.rename(columns={'rank_norm': f'rank_{i}'})
        resultado = resultado.merge(pred_rank, on='numero_de_cliente', how='inner')
        resultado['rank_score'] += resultado[f'rank_{i}'] * peso
        resultado = resultado.drop(columns=[f'rank_{i}'])
    
    # Ordenar por rank_score descendente
    resultado = resultado.sort_values('rank_score', ascending=False).reset_index(drop=True)
    resultado = resultado.rename(columns={'rank_score': 'prob'})
    
    logger.info(f"  ✓ Ensamble de {n_modelos} modelos completado")
    logger.info(f"  ✓ Total clientes: {len(resultado):,}")
    
    return resultado


def ensamble_votacion(predicciones_list, corte_base=11000):
    """
    Ensamble por VOTACIÓN
    Cuenta cuántos modelos predicen cada cliente en el top N
    
    Args:
        predicciones_list: Lista de DataFrames con columnas ['numero_de_cliente', 'prob']
        corte_base: Número de clientes a considerar en cada modelo
    
    Returns:
        DataFrame con predicciones ensambladas
    """
    logger.info("Estrategia: VOTACIÓN")
    logger.info(f"  Corte base: {corte_base}")
    
    n_modelos = len(predicciones_list)
    
    # Contar votos para cada cliente
    votos = {}
    
    for i, pred in enumerate(predicciones_list):
        # Top N clientes de este modelo
        top_clientes = pred.head(corte_base)['numero_de_cliente'].values
        
        for cliente in top_clientes:
            if cliente not in votos:
                votos[cliente] = 0
            votos[cliente] += 1
    
    # Crear DataFrame con votos
    resultado = pd.DataFrame({
        'numero_de_cliente': list(votos.keys()),
        'votos': list(votos.values())
    })
    
    # Ordenar por número de votos (descendente)
    resultado = resultado.sort_values('votos', ascending=False).reset_index(drop=True)
    
    # Calcular "probabilidad" como proporción de votos
    resultado['prob'] = resultado['votos'] / n_modelos
    
    logger.info(f"  ✓ Ensamble de {n_modelos} modelos completado")
    logger.info(f"  ✓ Clientes únicos: {len(resultado):,}")
    logger.info(f"  ✓ Clientes con todos los votos: {(resultado['votos'] == n_modelos).sum():,}")
    
    return resultado


def ensamble_max_prob(predicciones_list):
    """
    Ensamble por MÁXIMA PROBABILIDAD
    Toma la probabilidad más alta de cada cliente entre todos los modelos
    
    Args:
        predicciones_list: Lista de DataFrames con columnas ['numero_de_cliente', 'prob']
    
    Returns:
        DataFrame con predicciones ensambladas
    """
    logger.info("Estrategia: MÁXIMA PROBABILIDAD")
    
    # Crear DataFrame con todos los clientes
    resultado = predicciones_list[0][['numero_de_cliente']].copy()
    resultado['prob'] = predicciones_list[0]['prob']
    
    # Para cada modelo adicional, actualizar con el máximo
    for i, pred in enumerate(predicciones_list[1:], 1):
        pred_temp = pred[['numero_de_cliente', 'prob']].copy()
        pred_temp = pred_temp.rename(columns={'prob': f'prob_{i}'})
        resultado = resultado.merge(pred_temp, on='numero_de_cliente', how='inner')
        resultado['prob'] = resultado[['prob', f'prob_{i}']].max(axis=1)
        resultado = resultado.drop(columns=[f'prob_{i}'])
    
    # Ordenar por probabilidad descendente
    resultado = resultado.sort_values('prob', ascending=False).reset_index(drop=True)
    
    logger.info(f"  ✓ Ensamble completado")
    
    return resultado


# ============================================================================
# FUNCIÓN PRINCIPAL DE ENSAMBLE
# ============================================================================

def ensamblar_modelos(archivos_predicciones, estrategia='promedio', pesos=None, 
                      corte_base=11000, output_dir='./ensambles'):
    """
    Función principal para ensamblar predicciones de múltiples modelos
    
    Args:
        archivos_predicciones: Lista de rutas a archivos CSV con predicciones
        estrategia: 'promedio', 'rank_average', 'votacion', 'max_prob'
        pesos: Lista de pesos (solo para 'promedio' y 'rank_average')
        corte_base: Para estrategia 'votacion'
        output_dir: Directorio donde guardar resultados
    
    Returns:
        DataFrame con predicciones ensambladas
    """
    logger.info("="*80)
    logger.info("ENSAMBLADOR DE MODELOS")
    logger.info("="*80)
    logger.info(f"Estrategia: {estrategia.upper()}")
    logger.info(f"Modelos a ensamblar: {len(archivos_predicciones)}")
    
    # Cargar predicciones
    predicciones_list = []
    for i, archivo in enumerate(archivos_predicciones, 1):
        logger.info(f"  Cargando modelo {i}: {os.path.basename(archivo)}")
        df = pd.read_csv(archivo)
        
        # Verificar columnas necesarias
        if 'numero_de_cliente' not in df.columns or 'prob' not in df.columns:
            raise ValueError(f"El archivo {archivo} debe tener columnas 'numero_de_cliente' y 'prob'")
        
        predicciones_list.append(df[['numero_de_cliente', 'prob']])
    
    logger.info(f"\n✓ {len(predicciones_list)} modelos cargados\n")
    
    # Aplicar estrategia de ensamble
    if estrategia == 'promedio':
        resultado = ensamble_promedio_simple(predicciones_list, pesos)
    elif estrategia == 'rank_average':
        resultado = ensamble_rank_average(predicciones_list, pesos)
    elif estrategia == 'votacion':
        resultado = ensamble_votacion(predicciones_list, corte_base)
    elif estrategia == 'max_prob':
        resultado = ensamble_max_prob(predicciones_list)
    else:
        raise ValueError(f"Estrategia '{estrategia}' no reconocida")
    
    # Guardar resultado
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"ensamble_{estrategia}_{timestamp}.csv")
    resultado.to_csv(output_file, index=False)
    
    logger.info(f"\n✓ Ensamble guardado en: {output_file}")
    logger.info(f"✓ Total registros: {len(resultado):,}")
    logger.info(f"✓ Top 10 probabilidades: {resultado['prob'].head(10).values}")
    
    return resultado


def generar_submissions_ensamble(predicciones_ensamble, cortes, 
                                  experimento="ensamble", output_dir='./ensambles/kaggle'):
    """
    Genera archivos de submission para Kaggle a partir de predicciones ensambladas
    
    Args:
        predicciones_ensamble: DataFrame con predicciones ensambladas
        cortes: Lista de cortes a generar
        experimento: Nombre del experimento
        output_dir: Directorio donde guardar submissions
    """
    logger.info(f"\nGenerando {len(cortes)} submissions...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for corte in cortes:
        # Top N clientes
        clientes = predicciones_ensamble.head(corte)['numero_de_cliente']
        
        # Guardar SIN ENCABEZADO (formato Kaggle)
        filename = f"KA{experimento}_{corte}.csv"
        filepath = os.path.join(output_dir, filename)
        clientes.to_csv(filepath, index=False, header=False)
        
        if corte % 2500 == 0:
            logger.info(f"  Corte {corte}: {len(clientes)} envíos")
    
    logger.info(f"\n✓ {len(cortes)} archivos CSV generados en {output_dir}")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

def ejemplo_uso():
    """
    Ejemplo de cómo usar el ensamblador
    """
    print("="*80)
    print("EJEMPLO DE USO - ENSAMBLADOR DE MODELOS")
    print("="*80)
    
    # Rutas a las predicciones de diferentes modelos
    archivos = [
        "./exp/modelo_1/predicciones_final.csv",
        "./exp/modelo_2/predicciones_final.csv",
        "./exp/modelo_3/predicciones_final.csv",
    ]
    
    # Opción 1: PROMEDIO SIMPLE (pesos iguales)
    print("\n1. PROMEDIO SIMPLE:")
    resultado1 = ensamblar_modelos(
        archivos,
        estrategia='promedio'
    )
    
    # Opción 2: PROMEDIO PONDERADO (más peso al mejor modelo)
    print("\n2. PROMEDIO PONDERADO:")
    resultado2 = ensamblar_modelos(
        archivos,
        estrategia='promedio',
        pesos=[0.5, 0.3, 0.2]  # Modelo 1 tiene más peso
    )
    
    # Opción 3: RANK AVERAGE (más robusto)
    print("\n3. RANK AVERAGE:")
    resultado3 = ensamblar_modelos(
        archivos,
        estrategia='rank_average',
        pesos=[0.5, 0.3, 0.2]
    )
    
    # Opción 4: VOTACIÓN
    print("\n4. VOTACIÓN:")
    resultado4 = ensamblar_modelos(
        archivos,
        estrategia='votacion',
        corte_base=11000
    )
    
    # Opción 5: MÁXIMA PROBABILIDAD
    print("\n5. MÁXIMA PROBABILIDAD:")
    resultado5 = ensamblar_modelos(
        archivos,
        estrategia='max_prob'
    )
    
    # Generar submissions para el mejor ensamble
    cortes = [9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500]
    generar_submissions_ensamble(
        resultado3, 
        cortes, 
        experimento="ensamble_rank"
    )
    
    print("\n" + "="*80)
    print("✓ Todos los ensambles generados exitosamente")
    print("="*80)


# ============================================================================
# SCRIPT RÁPIDO PARA ENSAMBLAR 2 MODELOS
# ============================================================================

def ensamble_rapido_2_modelos(archivo1, archivo2, peso1=0.5, peso2=0.5):
    """
    Función rápida para ensamblar 2 modelos
    
    Ejemplo de uso:
        resultado = ensamble_rapido_2_modelos(
            'modelo1.csv', 
            'modelo2.csv', 
            peso1=0.6, 
            peso2=0.4
        )
    """
    logger.info("ENSAMBLE RÁPIDO - 2 MODELOS")
    
    # Cargar
    df1 = pd.read_csv(archivo1)
    df2 = pd.read_csv(archivo2)
    
    # Normalizar pesos
    suma = peso1 + peso2
    peso1, peso2 = peso1/suma, peso2/suma
    
    logger.info(f"  Pesos: {peso1:.2f} / {peso2:.2f}")
    
    # Merge
    resultado = df1[['numero_de_cliente', 'prob']].merge(
        df2[['numero_de_cliente', 'prob']], 
        on='numero_de_cliente', 
        suffixes=('_1', '_2')
    )
    
    # Promedio ponderado
    resultado['prob'] = (resultado['prob_1'] * peso1 + 
                        resultado['prob_2'] * peso2)
    
    resultado = resultado[['numero_de_cliente', 'prob']]
    resultado = resultado.sort_values('prob', ascending=False).reset_index(drop=True)
    
    logger.info(f"  ✓ Ensamble completado: {len(resultado):,} clientes")
    
    return resultado


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Descomentar para ver el ejemplo
    # ejemplo_uso()
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                  ENSAMBLADOR DE MODELOS                        ║
    ╚════════════════════════════════════════════════════════════════╝
    
    USO BÁSICO:
    
    # 1. Ensamblar 2 modelos (promedio simple)
    resultado = ensamble_rapido_2_modelos(
        'modelo1.csv', 
        'modelo2.csv'
    )
    
    # 2. Ensamblar 2 modelos (pesos diferentes)
    resultado = ensamble_rapido_2_modelos(
        'modelo1.csv', 
        'modelo2.csv',
        peso1=0.6,  # Modelo 1 tiene más peso
        peso2=0.4
    )
    
    # 3. Ensamblar múltiples modelos
    resultado = ensamblar_modelos(
        archivos=['modelo1.csv', 'modelo2.csv', 'modelo3.csv'],
        estrategia='promedio',  # o 'rank_average', 'votacion', 'max_prob'
        pesos=[0.5, 0.3, 0.2]
    )
    
    # 4. Generar submissions
    generar_submissions_ensamble(
        resultado,
        cortes=[9500, 10000, 10500, 11000, 11500, 12000],
        experimento="mi_ensamble"
    )
    
    ESTRATEGIAS DISPONIBLES:
    
    1. 'promedio'       → Promedio de probabilidades (clásico)
    2. 'rank_average'   → Promedio de rankings (más robusto)
    3. 'votacion'       → Cuenta cuántos modelos eligen cada cliente
    4. 'max_prob'       → Toma la probabilidad máxima
    
    RECOMENDACIÓN:
    - Para 2-3 modelos similares: 'promedio' con pesos iguales
    - Para modelos diversos: 'rank_average' con pesos basados en performance
    - Para muchos modelos: 'votacion'
    """)