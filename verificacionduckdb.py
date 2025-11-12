"""
Script para verificar el estado de la base de datos DuckDB
Muestra qué features ya están creadas y el estado general de los datos
"""

import duckdb
import pandas as pd
from datetime import datetime
from src.config import PATH_DATA_BASE_DB

def analizar_estado_database():
    """
    Analiza el estado completo de la base de datos DuckDB
    y muestra qué features de engineering ya están aplicadas
    """
    
    print("=" * 80)
    print("🔍 ANÁLISIS DE ESTADO DE LA BASE DE DATOS")
    print("=" * 80)
    print(f"Archivo: {PATH_DATA_BASE_DB}")
    print(f"Fecha análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    
    # ============================================
    # 1. TABLAS DISPONIBLES
    # ============================================
    
    print("\n📋 TABLAS DISPONIBLES:")
    print("-" * 80)
    
    tablas = conn.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main'
    """).df()
    
    if len(tablas) == 0:
        print("❌ No se encontraron tablas en la base de datos")
        conn.close()
        return
    
    for tabla in tablas['table_name']:
        count = conn.execute(f"SELECT COUNT(*) as cnt FROM {tabla}").fetchone()[0]
        print(f"   ✓ {tabla}: {count:,} registros")
    
    # Trabajar con la tabla principal
    tabla_principal = 'df_completo' if 'df_completo' in tablas['table_name'].values else tablas['table_name'].iloc[0]
    print(f"\n🎯 Analizando tabla principal: {tabla_principal}")
    
    # ============================================
    # 2. INFORMACIÓN GENERAL
    # ============================================
    
    print("\n" + "=" * 80)
    print("📊 INFORMACIÓN GENERAL")
    print("=" * 80)
    
    # Total registros
    total_registros = conn.execute(f"SELECT COUNT(*) FROM {tabla_principal}").fetchone()[0]
    print(f"Total registros: {total_registros:,}")
    
    # Total columnas
    columnas = conn.execute(f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = '{tabla_principal}'
    """).df()
    
    total_columnas = len(columnas)
    print(f"Total columnas: {total_columnas}")
    
    # foto_mes únicos
    foto_mes_info = conn.execute(f"""
        SELECT 
            COUNT(DISTINCT foto_mes) as meses_unicos,
            MIN(foto_mes) as mes_min,
            MAX(foto_mes) as mes_max
        FROM {tabla_principal}
    """).df()
    
    print(f"\nFoto_mes:")
    print(f"   - Meses únicos: {foto_mes_info['meses_unicos'][0]}")
    print(f"   - Primer mes: {foto_mes_info['mes_min'][0]}")
    print(f"   - Último mes: {foto_mes_info['mes_max'][0]}")
    
    # Distribución por mes
    print(f"\n   Distribución de registros por mes:")
    dist_meses = conn.execute(f"""
        SELECT foto_mes, COUNT(*) as registros
        FROM {tabla_principal}
        GROUP BY foto_mes
        ORDER BY foto_mes
    """).df()
    
    for _, row in dist_meses.iterrows():
        print(f"      {row['foto_mes']}: {row['registros']:>8,} registros")
    
    # Clientes únicos
    if 'numero_de_cliente' in columnas['column_name'].values:
        clientes_unicos = conn.execute(f"""
            SELECT COUNT(DISTINCT numero_de_cliente) as clientes
            FROM {tabla_principal}
        """).fetchone()[0]
        print(f"\nClientes únicos: {clientes_unicos:,}")
    
    # ============================================
    # 3. ANÁLISIS DE FEATURES CREADAS
    # ============================================
    
    print("\n" + "=" * 80)
    print("🔧 FEATURES DE ENGINEERING DETECTADAS")
    print("=" * 80)
    
    columnas_list = columnas['column_name'].tolist()
    
    # Detectar features temporales
    features = {
        'LAG': [col for col in columnas_list if '_lag_' in col],
        'DELTA': [col for col in columnas_list if '_delta_' in col],
        'SLOPE': [col for col in columnas_list if '_slope' in col or 'slope_' in col],
        'MAX': [col for col in columnas_list if '_max' in col or col.startswith('max_')],
        'MIN': [col for col in columnas_list if '_min' in col or col.startswith('min_')],
        'PERCENTIL': [col for col in columnas_list if '_percentil' in col],
        'RATIO': [col for col in columnas_list if '_ratio' in col or 'ratio_' in col],
        'CORREGIDA': [col for col in columnas_list if '_corregida' in col],
        'SUMA': [col for col in columnas_list if col.startswith('suma_') or col.startswith('total_') or col.startswith('monto_')],
    }
    
    for feature_type, cols in features.items():
        if cols:
            print(f"\n✅ {feature_type}: {len(cols)} columnas")
            
            # Mostrar algunos ejemplos
            ejemplos = cols[:5]
            for ejemplo in ejemplos:
                print(f"      - {ejemplo}")
            
            if len(cols) > 5:
                print(f"      ... y {len(cols) - 5} más")
            
            # Para LAG y DELTA, detectar ventana máxima
            if feature_type in ['LAG', 'DELTA']:
                max_ventana = 0
                for col in cols:
                    partes = col.split('_')
                    try:
                        num = int(partes[-1])
                        max_ventana = max(max_ventana, num)
                    except:
                        pass
                if max_ventana > 0:
                    print(f"      → Ventana máxima detectada: {max_ventana}")
        else:
            print(f"\n❌ {feature_type}: No creadas")
    
    # ============================================
    # 4. COLUMNAS ORIGINALES VS DERIVADAS
    # ============================================
    
    print("\n" + "=" * 80)
    print("📈 RESUMEN DE COLUMNAS")
    print("=" * 80)
    
    # Clasificar columnas
    cols_originales = []
    cols_derivadas = []
    
    sufijos_derivados = ['_lag_', '_delta_', '_slope', '_max', '_min', 
                         '_percentil', '_ratio', '_corregida', 'suma_', 
                         'total_', 'monto_', 'ganancia_']
    
    for col in columnas_list:
        if any(sufijo in col for sufijo in sufijos_derivados):
            cols_derivadas.append(col)
        else:
            cols_originales.append(col)
    
    print(f"Columnas originales: {len(cols_originales)}")
    print(f"Columnas derivadas (FE): {len(cols_derivadas)}")
    print(f"Total: {len(cols_originales) + len(cols_derivadas)}")
    
    # ============================================
    # 5. VERIFICACIÓN DE NULOS
    # ============================================
    
    print("\n" + "=" * 80)
    print("🔍 ANÁLISIS DE VALORES NULOS (columnas clave)")
    print("=" * 80)
    
    columnas_clave = ['numero_de_cliente', 'foto_mes', 'clase_ternaria']
    
    for col in columnas_clave:
        if col in columnas_list:
            nulls = conn.execute(f"""
                SELECT COUNT(*) as nulls
                FROM {tabla_principal}
                WHERE {col} IS NULL
            """).fetchone()[0]
            
            pct_nulls = (nulls / total_registros) * 100 if total_registros > 0 else 0
            
            if nulls == 0:
                print(f"✅ {col}: Sin nulos")
            else:
                print(f"⚠️  {col}: {nulls:,} nulos ({pct_nulls:.2f}%)")
    
    # ============================================
    # 6. ESTADÍSTICAS DE TIPOS DE DATOS
    # ============================================
    
    print("\n" + "=" * 80)
    print("📊 TIPOS DE DATOS")
    print("=" * 80)
    
    tipos = columnas.groupby('data_type').size().sort_values(ascending=False)
    
    for tipo, cantidad in tipos.items():
        print(f"   {tipo}: {cantidad} columnas")
    
    # ============================================
    # 7. DETECCIÓN DE FEATURES FALTANTES
    # ============================================
    
    print("\n" + "=" * 80)
    print("⚠️  FEATURES QUE PODRÍAN ESTAR FALTANDO")
    print("=" * 80)
    
    checks = {
        'LAG': len(features['LAG']) > 0,
        'DELTA': len(features['DELTA']) > 0,
        'SLOPE': len(features['SLOPE']) > 0,
        'MAX/MIN': len(features['MAX']) > 0 or len(features['MIN']) > 0,
        'PERCENTIL': len(features['PERCENTIL']) > 0,
        'RATIO': len(features['RATIO']) > 0,
    }
    
    faltantes = [feat for feat, existe in checks.items() if not existe]
    
    if faltantes:
        print("Podrían estar faltando:")
        for feat in faltantes:
            print(f"   ❌ {feat}")
    else:
        print("✅ Todas las features principales parecen estar creadas")
    
    # ============================================
    # 8. RECOMENDACIONES
    # ============================================
    
    print("\n" + "=" * 80)
    print("💡 RECOMENDACIONES")
    print("=" * 80)
    
    if total_columnas < 300:
        print("⚠️  Pocas columnas detectadas. Posiblemente el FE no está completo.")
    elif total_columnas > 2000:
        print("✅ Muchas columnas detectadas. El FE parece estar bastante avanzado.")
    else:
        print("ℹ️  Cantidad moderada de columnas. Verifica las features faltantes arriba.")
    
    if len(features['LAG']) > 0:
        print(f"✅ Se detectaron LAGs. Ventana procesada.")
    
    if len(features['PERCENTIL']) > 0:
        print(f"✅ Se detectaron percentiles. Normalización aplicada.")
    
    # ============================================
    # RESUMEN FINAL
    # ============================================
    
    print("\n" + "=" * 80)
    print("📋 RESUMEN EJECUTIVO")
    print("=" * 80)
    print(f"✓ Registros: {total_registros:,}")
    print(f"✓ Columnas totales: {total_columnas}")
    print(f"✓ Columnas originales: {len(cols_originales)}")
    print(f"✓ Columnas derivadas: {len(cols_derivadas)}")
    print(f"✓ Meses disponibles: {foto_mes_info['meses_unicos'][0]} ({foto_mes_info['mes_min'][0]} - {foto_mes_info['mes_max'][0]})")
    print(f"✓ Features aplicadas: {sum(1 for v in checks.values() if v)}/{len(checks)}")
    print("=" * 80)
    
    conn.close()


if __name__ == "__main__":
    try:
        analizar_estado_database()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()