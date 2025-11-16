"""
EDA Temporal y Data Drift Analysis
Análisis exploratorio enfocado en:
- Evolución temporal de variables
- Data drift por mes (especial atención en agosto)
- Tendencias de bajas
- Variables problemáticas para eliminar
- Registros por mes

Autor: Data Scientist Junior
Fecha: 2025-11-16
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Configuración de gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DATASET_PATH = "~/datasets/competencia_02_crudo.csv.gz"
OUTPUT_DIR = "./eda_temporal"

# Meses de interés especial
MESES_TRAIN = [201901, 201902, 201903, 201904, 201905, 201906,
               201907, 201908, 201909, 201910, 201911, 201912,
               202001, 202002, 202003, 202004, 202005, 202006,
               202007, 202008, 202009, 202010, 202011, 202012,
               202101, 202102]
MESES_TEST = [202104, 202106]
MES_FINAL = 202108  # Especial atención aquí

# Umbrales para detección de problemas
UMBRAL_MISSING_VARIABLE = 0.95  # Si >95% missing, marcar
UMBRAL_DRIFT_PSI = 0.25  # PSI > 0.25 indica drift significativo
UMBRAL_VARIANZA_CERO = 0.01  # Si varianza < 0.01, poco informativa

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def crear_directorio(path):
    """Crea un directorio si no existe"""
    os.makedirs(path, exist_ok=True)


def calcular_clase_ternaria(df):
    """Calcula la clase_ternaria según la permanencia del cliente"""
    print("Calculando clase_ternaria...")
    
    df['periodo0'] = (df['foto_mes'] // 100) * 12 + (df['foto_mes'] % 100)
    df = df.sort_values(['numero_de_cliente', 'periodo0']).reset_index(drop=True)
    
    df['periodo1'] = df.groupby('numero_de_cliente')['periodo0'].shift(-1)
    df['periodo2'] = df.groupby('numero_de_cliente')['periodo0'].shift(-2)
    
    periodo_ultimo = df['periodo0'].max()
    periodo_anteultimo = periodo_ultimo - 1
    
    df['clase_ternaria'] = 'CONTINUA'
    
    mask_baja1 = (df['periodo0'] < periodo_ultimo) & \
                 (df['periodo1'].isna() | (df['periodo0'] + 1 < df['periodo1']))
    df.loc[mask_baja1, 'clase_ternaria'] = 'BAJA+1'
    
    mask_baja2 = (df['periodo0'] < periodo_anteultimo) & \
                 (df['periodo0'] + 1 == df['periodo1']) & \
                 (df['periodo2'].isna() | (df['periodo0'] + 2 < df['periodo2']))
    df.loc[mask_baja2, 'clase_ternaria'] = 'BAJA+2'
    
    df = df.drop(['periodo0', 'periodo1', 'periodo2'], axis=1)
    
    return df


# ============================================================================
# ANÁLISIS 1: REGISTROS Y CLASES POR MES
# ============================================================================

def analizar_registros_por_mes(df):
    """
    Analiza cantidad de registros y distribución de clases por mes
    """
    print("\n" + "="*80)
    print("ANÁLISIS 1: REGISTROS Y CLASES POR MES")
    print("="*80)
    
    # Contar registros por mes
    registros_por_mes = df.groupby('foto_mes').size().reset_index(name='total_registros')
    
    # Contar clases por mes
    clases_por_mes = df.groupby(['foto_mes', 'clase_ternaria']).size().unstack(fill_value=0)
    clases_por_mes = clases_por_mes.reset_index()
    
    # Calcular porcentajes
    clases_por_mes['total'] = clases_por_mes[['CONTINUA', 'BAJA+1', 'BAJA+2']].sum(axis=1)
    clases_por_mes['pct_continua'] = 100 * clases_por_mes['CONTINUA'] / clases_por_mes['total']
    clases_por_mes['pct_baja1'] = 100 * clases_por_mes['BAJA+1'] / clases_por_mes['total']
    clases_por_mes['pct_baja2'] = 100 * clases_por_mes['BAJA+2'] / clases_por_mes['total']
    clases_por_mes['pct_total_bajas'] = clases_por_mes['pct_baja1'] + clases_por_mes['pct_baja2']
    
    # Mostrar resumen
    print("\nResumen de registros por mes:")
    print(clases_por_mes[['foto_mes', 'total', 'CONTINUA', 'BAJA+1', 'BAJA+2', 
                           'pct_continua', 'pct_baja1', 'pct_baja2', 'pct_total_bajas']].to_string(index=False))
    
    # Identificar mes de agosto
    print(f"\n{'='*80}")
    print(f"ATENCIÓN ESPECIAL: MES {MES_FINAL} (Agosto)")
    print(f"{'='*80}")
    agosto = clases_por_mes[clases_por_mes['foto_mes'] == MES_FINAL]
    if not agosto.empty:
        print(f"Total registros: {agosto['total'].values[0]:,}")
        print(f"CONTINUA: {agosto['CONTINUA'].values[0]:,} ({agosto['pct_continua'].values[0]:.2f}%)")
        print(f"BAJA+1: {agosto['BAJA+1'].values[0]:,} ({agosto['pct_baja1'].values[0]:.2f}%)")
        print(f"BAJA+2: {agosto['BAJA+2'].values[0]:,} ({agosto['pct_baja2'].values[0]:.2f}%)")
    
    # Graficar
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis de Registros y Clases por Mes', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Total de registros por mes
    axes[0, 0].plot(clases_por_mes['foto_mes'], clases_por_mes['total'], 
                    marker='o', linewidth=2, markersize=6, color='steelblue')
    axes[0, 0].axvline(x=MES_FINAL, color='red', linestyle='--', alpha=0.7, label=f'Mes {MES_FINAL}')
    axes[0, 0].set_title('Total de Registros por Mes', fontweight='bold')
    axes[0, 0].set_xlabel('Mes')
    axes[0, 0].set_ylabel('Cantidad de Registros')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Gráfico 2: Distribución de clases (números absolutos)
    axes[0, 1].plot(clases_por_mes['foto_mes'], clases_por_mes['CONTINUA'], 
                    marker='o', label='CONTINUA', linewidth=2)
    axes[0, 1].plot(clases_por_mes['foto_mes'], clases_por_mes['BAJA+1'], 
                    marker='s', label='BAJA+1', linewidth=2)
    axes[0, 1].plot(clases_por_mes['foto_mes'], clases_por_mes['BAJA+2'], 
                    marker='^', label='BAJA+2', linewidth=2)
    axes[0, 1].axvline(x=MES_FINAL, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Distribución de Clases por Mes (Absoluto)', fontweight='bold')
    axes[0, 1].set_xlabel('Mes')
    axes[0, 1].set_ylabel('Cantidad')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Gráfico 3: Porcentaje de bajas totales
    axes[1, 0].plot(clases_por_mes['foto_mes'], clases_por_mes['pct_total_bajas'], 
                    marker='o', linewidth=3, markersize=8, color='darkred')
    axes[1, 0].axvline(x=MES_FINAL, color='red', linestyle='--', alpha=0.7, label=f'Mes {MES_FINAL}')
    axes[1, 0].axhline(y=clases_por_mes['pct_total_bajas'].mean(), 
                       color='gray', linestyle=':', alpha=0.7, label='Promedio')
    axes[1, 0].set_title('Tendencia de Bajas Totales (BAJA+1 + BAJA+2)', fontweight='bold')
    axes[1, 0].set_xlabel('Mes')
    axes[1, 0].set_ylabel('% Bajas')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Gráfico 4: Distribución porcentual
    axes[1, 1].plot(clases_por_mes['foto_mes'], clases_por_mes['pct_baja1'], 
                    marker='s', label='BAJA+1', linewidth=2)
    axes[1, 1].plot(clases_por_mes['foto_mes'], clases_por_mes['pct_baja2'], 
                    marker='^', label='BAJA+2', linewidth=2)
    axes[1, 1].axvline(x=MES_FINAL, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('Distribución Porcentual de Bajas', fontweight='bold')
    axes[1, 1].set_xlabel('Mes')
    axes[1, 1].set_ylabel('Porcentaje (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_registros_y_clases.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Gráfico guardado: 01_registros_y_clases.png")
    
    return clases_por_mes


# ============================================================================
# ANÁLISIS 2: MISSINGS POR MES
# ============================================================================

def analizar_missings_por_mes(df):
    """
    Analiza evolución de valores faltantes por mes
    """
    print("\n" + "="*80)
    print("ANÁLISIS 2: MISSINGS POR MES")
    print("="*80)
    
    # Excluir columnas de ID
    cols_analizar = [col for col in df.columns 
                     if col not in ['numero_de_cliente', 'foto_mes', 'clase_ternaria']]
    
    # Calcular % de missings por mes y variable
    missing_por_mes = []
    
    for mes in sorted(df['foto_mes'].unique()):
        df_mes = df[df['foto_mes'] == mes]
        n_registros = len(df_mes)
        
        for col in cols_analizar:
            n_missing = df_mes[col].isna().sum()
            pct_missing = 100 * n_missing / n_registros
            
            missing_por_mes.append({
                'foto_mes': mes,
                'variable': col,
                'n_missing': n_missing,
                'pct_missing': pct_missing
            })
    
    df_missing = pd.DataFrame(missing_por_mes)
    
    # Variables con más de UMBRAL_MISSING_VARIABLE% missings en algún mes
    vars_problematicas = df_missing[df_missing['pct_missing'] > UMBRAL_MISSING_VARIABLE * 100]['variable'].unique()
    
    print(f"\nVariables con >{UMBRAL_MISSING_VARIABLE*100}% missings en algún mes:")
    if len(vars_problematicas) > 0:
        for var in vars_problematicas:
            var_data = df_missing[df_missing['variable'] == var]
            max_missing = var_data['pct_missing'].max()
            mes_max = var_data[var_data['pct_missing'] == max_missing]['foto_mes'].values[0]
            print(f"  - {var}: {max_missing:.1f}% en mes {mes_max}")
        
        # Guardar lista
        pd.DataFrame({'variable': vars_problematicas}).to_csv(
            os.path.join(OUTPUT_DIR, 'variables_con_muchos_missings.csv'), index=False
        )
        print(f"\n✓ Lista guardada: variables_con_muchos_missings.csv")
    else:
        print("  ✓ No hay variables con missings extremos")
    
    # Análisis específico para agosto
    print(f"\nMissings en mes {MES_FINAL}:")
    agosto_missing = df_missing[df_missing['foto_mes'] == MES_FINAL]
    top_missing_agosto = agosto_missing.nlargest(10, 'pct_missing')[['variable', 'pct_missing']]
    print(top_missing_agosto.to_string(index=False))
    
    # Graficar evolución de top variables con missings
    top_vars_missing = df_missing.groupby('variable')['pct_missing'].max().nlargest(10).index
    
    if len(top_vars_missing) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for var in top_vars_missing:
            var_data = df_missing[df_missing['variable'] == var]
            ax.plot(var_data['foto_mes'], var_data['pct_missing'], 
                   marker='o', label=var, linewidth=2, markersize=4)
        
        ax.axvline(x=MES_FINAL, color='red', linestyle='--', alpha=0.7, label=f'Mes {MES_FINAL}')
        ax.axhline(y=UMBRAL_MISSING_VARIABLE * 100, color='red', linestyle=':', 
                  alpha=0.5, label=f'Umbral {UMBRAL_MISSING_VARIABLE*100}%')
        ax.set_title('Evolución de Missings - Top 10 Variables', fontsize=14, fontweight='bold')
        ax.set_xlabel('Mes')
        ax.set_ylabel('% Missing')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '02_evolucion_missings.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Gráfico guardado: 02_evolucion_missings.png")
    
    return df_missing, vars_problematicas


# ============================================================================
# ANÁLISIS 3: DATA DRIFT (PSI - Population Stability Index)
# ============================================================================

def calcular_psi(esperado, actual, bins=10):
    """
    Calcula Population Stability Index (PSI)
    PSI < 0.1: Sin cambio significativo
    PSI 0.1-0.25: Cambio moderado
    PSI > 0.25: Cambio significativo
    """
    # Eliminar NaNs
    esperado_clean = esperado[~np.isnan(esperado)]
    actual_clean = actual[~np.isnan(actual)]
    
    if len(esperado_clean) == 0 or len(actual_clean) == 0:
        return np.nan
    
    # Calcular percentiles en datos esperados
    try:
        percentiles = np.percentile(esperado_clean, np.linspace(0, 100, bins + 1))
        percentiles[0] = -np.inf
        percentiles[-1] = np.inf
        
        # Contar en cada bin
        esperado_counts = np.histogram(esperado_clean, bins=percentiles)[0]
        actual_counts = np.histogram(actual_clean, bins=percentiles)[0]
        
        # Evitar divisiones por cero
        esperado_pct = esperado_counts / len(esperado_clean)
        actual_pct = actual_counts / len(actual_clean)
        
        esperado_pct = np.where(esperado_pct == 0, 0.0001, esperado_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        # Calcular PSI
        psi = np.sum((actual_pct - esperado_pct) * np.log(actual_pct / esperado_pct))
        
        return psi
    except:
        return np.nan


def analizar_data_drift(df):
    """
    Analiza data drift comparando agosto con meses anteriores
    """
    print("\n" + "="*80)
    print("ANÁLISIS 3: DATA DRIFT (Population Stability Index)")
    print("="*80)
    
    # Mes de referencia: último mes de train antes de agosto
    mes_referencia = 202106
    
    print(f"\nComparando mes {MES_FINAL} (agosto) vs mes {mes_referencia} (referencia)")
    print(f"Umbral PSI: {UMBRAL_DRIFT_PSI} (>0.25 = drift significativo)")
    
    # Filtrar datos
    df_ref = df[df['foto_mes'] == mes_referencia]
    df_ago = df[df['foto_mes'] == MES_FINAL]
    
    # Variables numéricas
    cols_numericas = df.select_dtypes(include=[np.number]).columns
    cols_analizar = [col for col in cols_numericas 
                     if col not in ['numero_de_cliente', 'foto_mes']]
    
    # Calcular PSI para cada variable
    psi_results = []
    
    print("\nCalculando PSI para cada variable...")
    for col in cols_analizar:
        psi = calcular_psi(df_ref[col].values, df_ago[col].values)
        
        if not np.isnan(psi):
            # Calcular también diferencia de medias
            mean_ref = df_ref[col].mean()
            mean_ago = df_ago[col].mean()
            diff_mean = mean_ago - mean_ref
            pct_change = 100 * diff_mean / mean_ref if mean_ref != 0 else np.nan
            
            psi_results.append({
                'variable': col,
                'psi': psi,
                'mean_ref': mean_ref,
                'mean_ago': mean_ago,
                'diff_mean': diff_mean,
                'pct_change': pct_change,
                'drift_level': 'Alto' if psi > 0.25 else ('Moderado' if psi > 0.1 else 'Bajo')
            })
    
    df_psi = pd.DataFrame(psi_results).sort_values('psi', ascending=False)
    
    # Variables con drift alto
    vars_drift_alto = df_psi[df_psi['psi'] > UMBRAL_DRIFT_PSI]
    
    print(f"\n{'='*80}")
    print(f"VARIABLES CON DRIFT ALTO (PSI > {UMBRAL_DRIFT_PSI}):")
    print(f"{'='*80}")
    
    if len(vars_drift_alto) > 0:
        print(f"\nTotal: {len(vars_drift_alto)} variables")
        print("\nTop 20 variables con mayor drift:")
        print(vars_drift_alto[['variable', 'psi', 'pct_change', 'drift_level']].head(20).to_string(index=False))
        
        # Guardar lista completa
        vars_drift_alto.to_csv(
            os.path.join(OUTPUT_DIR, 'variables_con_drift_alto.csv'), index=False
        )
        print(f"\n✓ Lista completa guardada: variables_con_drift_alto.csv")
    else:
        print("  ✓ No hay variables con drift alto")
    
    # Guardar PSI de todas las variables
    df_psi.to_csv(os.path.join(OUTPUT_DIR, 'psi_todas_variables.csv'), index=False)
    print(f"✓ PSI de todas las variables: psi_todas_variables.csv")
    
    # Graficar top variables con drift
    top_n = min(20, len(df_psi))
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Data Drift: Mes {MES_FINAL} vs Mes {mes_referencia}', fontsize=16, fontweight='bold')
    
    # Gráfico 1: PSI values
    top_psi = df_psi.head(top_n)
    colors = ['red' if x > UMBRAL_DRIFT_PSI else 'orange' if x > 0.1 else 'green' 
              for x in top_psi['psi']]
    
    axes[0].barh(range(len(top_psi)), top_psi['psi'], color=colors)
    axes[0].set_yticks(range(len(top_psi)))
    axes[0].set_yticklabels(top_psi['variable'], fontsize=8)
    axes[0].axvline(x=UMBRAL_DRIFT_PSI, color='red', linestyle='--', 
                    label=f'Umbral {UMBRAL_DRIFT_PSI}')
    axes[0].axvline(x=0.1, color='orange', linestyle=':', label='Umbral 0.1')
    axes[0].set_xlabel('PSI Value')
    axes[0].set_title(f'Top {top_n} Variables con Mayor Drift', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    
    # Gráfico 2: Cambio porcentual en media
    top_psi_sorted = top_psi.sort_values('pct_change')
    colors2 = ['red' if abs(x) > 50 else 'orange' if abs(x) > 20 else 'green' 
               for x in top_psi_sorted['pct_change']]
    
    axes[1].barh(range(len(top_psi_sorted)), top_psi_sorted['pct_change'], color=colors2)
    axes[1].set_yticks(range(len(top_psi_sorted)))
    axes[1].set_yticklabels(top_psi_sorted['variable'], fontsize=8)
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_xlabel('Cambio % en Media')
    axes[1].set_title('Cambio Porcentual en Media', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_data_drift_psi.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Gráfico guardado: 03_data_drift_psi.png")
    
    return df_psi, vars_drift_alto


# ============================================================================
# ANÁLISIS 4: EVOLUCIÓN DE VARIABLES CLAVE
# ============================================================================

def analizar_evolucion_variables(df, n_vars=10):
    """
    Analiza la evolución temporal de las variables más importantes
    """
    print("\n" + "="*80)
    print(f"ANÁLISIS 4: EVOLUCIÓN TEMPORAL DE TOP {n_vars} VARIABLES")
    print("="*80)
    
    # Variables numéricas
    cols_numericas = df.select_dtypes(include=[np.number]).columns
    cols_analizar = [col for col in cols_numericas 
                     if col not in ['numero_de_cliente', 'foto_mes']]
    
    # Calcular varianza de cada variable (promedio a través de los meses)
    variances = []
    for col in cols_analizar:
        var = df[col].var()
        if not np.isnan(var) and var > 0:
            variances.append((col, var))
    
    # Top variables por varianza (son las que más varían = más informativas)
    top_vars = sorted(variances, key=lambda x: x[1], reverse=True)[:n_vars]
    top_var_names = [v[0] for v in top_vars]
    
    print(f"\nTop {n_vars} variables por varianza:")
    for i, (var, variance) in enumerate(top_vars, 1):
        print(f"  {i}. {var}: {variance:.2e}")
    
    # Calcular media y std por mes para estas variables
    evolucion = []
    
    for mes in sorted(df['foto_mes'].unique()):
        df_mes = df[df['foto_mes'] == mes]
        
        for var in top_var_names:
            evolucion.append({
                'foto_mes': mes,
                'variable': var,
                'mean': df_mes[var].mean(),
                'std': df_mes[var].std(),
                'min': df_mes[var].min(),
                'max': df_mes[var].max(),
                'median': df_mes[var].median()
            })
    
    df_evolucion = pd.DataFrame(evolucion)
    
    # Graficar evolución
    n_cols = 2
    n_rows = (n_vars + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    fig.suptitle('Evolución Temporal de Variables Clave', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, var in enumerate(top_var_names):
        var_data = df_evolucion[df_evolucion['variable'] == var]
        
        axes[i].plot(var_data['foto_mes'], var_data['mean'], 
                    marker='o', linewidth=2, label='Media')
        axes[i].fill_between(var_data['foto_mes'], 
                            var_data['mean'] - var_data['std'],
                            var_data['mean'] + var_data['std'],
                            alpha=0.3, label='±1 std')
        axes[i].axvline(x=MES_FINAL, color='red', linestyle='--', 
                       alpha=0.7, label=f'Mes {MES_FINAL}')
        axes[i].set_title(var, fontweight='bold', fontsize=10)
        axes[i].set_xlabel('Mes', fontsize=8)
        axes[i].set_ylabel('Valor', fontsize=8)
        axes[i].legend(fontsize=7)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45, labelsize=7)
        axes[i].tick_params(axis='y', labelsize=7)
    
    # Ocultar ejes vacíos si n_vars es impar
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_evolucion_variables_clave.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Gráfico guardado: 04_evolucion_variables_clave.png")
    
    return df_evolucion, top_var_names


# ============================================================================
# ANÁLISIS 5: VARIABLES CON VARIANZA CERO O MUY BAJA
# ============================================================================

def analizar_variables_constantes(df):
    """
    Identifica variables con varianza cero o muy baja (poco informativas)
    """
    print("\n" + "="*80)
    print("ANÁLISIS 5: VARIABLES CONSTANTES O POCO INFORMATIVAS")
    print("="*80)
    
    # Variables numéricas
    cols_numericas = df.select_dtypes(include=[np.number]).columns
    cols_analizar = [col for col in cols_numericas 
                     if col not in ['numero_de_cliente', 'foto_mes']]
    
    # Calcular varianza por mes
    vars_constantes_por_mes = []
    
    for mes in sorted(df['foto_mes'].unique()):
        df_mes = df[df['foto_mes'] == mes]
        
        for col in cols_analizar:
            var = df_mes[col].var()
            n_unique = df_mes[col].nunique()
            
            vars_constantes_por_mes.append({
                'foto_mes': mes,
                'variable': col,
                'varianza': var,
                'n_unique': n_unique
            })
    
    df_vars = pd.DataFrame(vars_constantes_por_mes)
    
    # Variables con varianza muy baja en agosto
    agosto_vars = df_vars[df_vars['foto_mes'] == MES_FINAL]
    vars_baja_varianza = agosto_vars[agosto_vars['varianza'] < UMBRAL_VARIANZA_CERO]
    
    print(f"\nVariables con varianza < {UMBRAL_VARIANZA_CERO} en mes {MES_FINAL}:")
    if len(vars_baja_varianza) > 0:
        print(f"Total: {len(vars_baja_varianza)} variables")
        print("\nTop 20:")
        print(vars_baja_varianza[['variable', 'varianza', 'n_unique']].head(20).to_string(index=False))
        
        # Guardar lista
        vars_baja_varianza.to_csv(
            os.path.join(OUTPUT_DIR, 'variables_baja_varianza.csv'), index=False
        )
        print(f"\n✓ Lista guardada: variables_baja_varianza.csv")
    else:
        print("  ✓ No hay variables con varianza extremadamente baja")
    
    # Variables constantes en todos los meses
    vars_siempre_constantes = []
    for var in cols_analizar:
        var_data = df_vars[df_vars['variable'] == var]
        if (var_data['varianza'] < UMBRAL_VARIANZA_CERO).all():
            vars_siempre_constantes.append(var)
    
    if len(vars_siempre_constantes) > 0:
        print(f"\nVariables SIEMPRE constantes (en todos los meses):")
        print(f"Total: {len(vars_siempre_constantes)} variables")
        for var in vars_siempre_constantes[:20]:
            print(f"  - {var}")
        
        # Guardar
        pd.DataFrame({'variable': vars_siempre_constantes}).to_csv(
            os.path.join(OUTPUT_DIR, 'variables_siempre_constantes.csv'), index=False
        )
        print(f"\n✓ Lista guardada: variables_siempre_constantes.csv")
    
    return df_vars, vars_baja_varianza, vars_siempre_constantes


# ============================================================================
# REPORTE FINAL
# ============================================================================

def generar_reporte_final(clases_por_mes, vars_problematicas_missing, 
                         vars_drift_alto, vars_baja_varianza, vars_siempre_constantes):
    """
    Genera un reporte resumen con recomendaciones
    """
    print("\n" + "="*80)
    print("REPORTE FINAL Y RECOMENDACIONES")
    print("="*80)
    
    reporte = []
    
    # Sección 1: Registros y Clases
    reporte.append("="*80)
    reporte.append("1. ANÁLISIS DE REGISTROS Y CLASES")
    reporte.append("="*80)
    
    agosto = clases_por_mes[clases_por_mes['foto_mes'] == MES_FINAL]
    if not agosto.empty:
        reporte.append(f"\nMes {MES_FINAL} (Predicción final):")
        reporte.append(f"  - Total registros: {agosto['total'].values[0]:,}")
        reporte.append(f"  - CONTINUA: {agosto['pct_continua'].values[0]:.2f}%")
        reporte.append(f"  - BAJA+1: {agosto['pct_baja1'].values[0]:.2f}%")
        reporte.append(f"  - BAJA+2: {agosto['pct_baja2'].values[0]:.2f}%")
        reporte.append(f"  - Total Bajas: {agosto['pct_total_bajas'].values[0]:.2f}%")
    
    # Tendencia de bajas
    promedio_bajas = clases_por_mes['pct_total_bajas'].mean()
    bajas_agosto = agosto['pct_total_bajas'].values[0] if not agosto.empty else 0
    
    reporte.append(f"\nTendencia de bajas:")
    reporte.append(f"  - Promedio histórico: {promedio_bajas:.2f}%")
    reporte.append(f"  - Mes {MES_FINAL}: {bajas_agosto:.2f}%")
    if bajas_agosto > promedio_bajas:
        reporte.append(f"  ⚠️  ALERTA: Mes {MES_FINAL} tiene MÁS bajas que el promedio (+{bajas_agosto - promedio_bajas:.2f}%)")
    else:
        reporte.append(f"  ✓ Mes {MES_FINAL} tiene menos bajas que el promedio")
    
    # Sección 2: Missings
    reporte.append("\n" + "="*80)
    reporte.append("2. VARIABLES CON MUCHOS MISSINGS")
    reporte.append("="*80)
    
    if len(vars_problematicas_missing) > 0:
        reporte.append(f"\n⚠️  {len(vars_problematicas_missing)} variables con >{UMBRAL_MISSING_VARIABLE*100}% missings")
        reporte.append("\nRECOMENDACIÓN: Considerar ELIMINAR estas variables:")
        for var in vars_problematicas_missing[:20]:
            reporte.append(f"  - {var}")
        if len(vars_problematicas_missing) > 20:
            reporte.append(f"  ... y {len(vars_problematicas_missing) - 20} más")
    else:
        reporte.append("\n✓ No hay variables con missings extremos")
    
    # Sección 3: Data Drift
    reporte.append("\n" + "="*80)
    reporte.append("3. VARIABLES CON DATA DRIFT ALTO")
    reporte.append("="*80)
    
    if len(vars_drift_alto) > 0:
        reporte.append(f"\n⚠️  {len(vars_drift_alto)} variables con drift significativo (PSI > {UMBRAL_DRIFT_PSI})")
        reporte.append(f"\nEstas variables cambiaron mucho en mes {MES_FINAL}:")
        for _, row in vars_drift_alto.head(20).iterrows():
            reporte.append(f"  - {row['variable']}: PSI={row['psi']:.3f}, Cambio={row['pct_change']:.1f}%")
        if len(vars_drift_alto) > 20:
            reporte.append(f"  ... y {len(vars_drift_alto) - 20} más")
        
        reporte.append("\nRECOMENDACIÓN:")
        reporte.append("  1. MONITOREAR estas variables de cerca")
        reporte.append("  2. Considerar re-entrenar con datos más recientes")
        reporte.append("  3. Si el drift es muy extremo, considerar ELIMINAR la variable")
    else:
        reporte.append("\n✓ No hay variables con drift significativo")
    
    # Sección 4: Varianza Baja
    reporte.append("\n" + "="*80)
    reporte.append("4. VARIABLES CON VARIANZA MUY BAJA")
    reporte.append("="*80)
    
    if len(vars_siempre_constantes) > 0:
        reporte.append(f"\n⚠️  {len(vars_siempre_constantes)} variables constantes en TODOS los meses")
        reporte.append("\nRECOMENDACIÓN: ELIMINAR estas variables (no aportan información):")
        for var in vars_siempre_constantes[:20]:
            reporte.append(f"  - {var}")
        if len(vars_siempre_constantes) > 20:
            reporte.append(f"  ... y {len(vars_siempre_constantes) - 20} más")
    else:
        reporte.append("\n✓ No hay variables siempre constantes")
    
    if len(vars_baja_varianza) > 0:
        reporte.append(f"\nAdemás, {len(vars_baja_varianza)} variables con baja varianza en mes {MES_FINAL}")
        reporte.append("  (Ver archivo variables_baja_varianza.csv)")
    
    # Sección 5: Recomendaciones Finales
    reporte.append("\n" + "="*80)
    reporte.append("5. RECOMENDACIONES FINALES")
    reporte.append("="*80)
    
    # Contar variables problemáticas
    vars_eliminar = set()
    vars_eliminar.update(vars_problematicas_missing)
    vars_eliminar.update(vars_siempre_constantes)
    
    # Variables con drift extremo (PSI > 0.5)
    vars_drift_extremo = vars_drift_alto[vars_drift_alto['psi'] > 0.5]['variable'].tolist() if len(vars_drift_alto) > 0 else []
    
    reporte.append(f"\n📊 RESUMEN:")
    reporte.append(f"  - Variables a ELIMINAR (missings + constantes): {len(vars_eliminar)}")
    reporte.append(f"  - Variables con drift ALTO: {len(vars_drift_alto)}")
    reporte.append(f"  - Variables con drift EXTREMO (PSI>0.5): {len(vars_drift_extremo)}")
    
    reporte.append("\n✅ ACCIONES RECOMENDADAS:")
    
    reporte.append("\n1. ELIMINAR variables:")
    if len(vars_eliminar) > 0:
        reporte.append(f"   → {len(vars_eliminar)} variables (ver archivos CSV generados)")
        reporte.append("   → Esto reducirá overfitting y acelerará el entrenamiento")
    else:
        reporte.append("   → No hay variables para eliminar")
    
    reporte.append("\n2. MONITOREAR variables con drift:")
    if len(vars_drift_alto) > 0:
        reporte.append(f"   → {len(vars_drift_alto)} variables cambiaron significativamente")
        reporte.append("   → Considerar entrenar con ventana temporal más corta")
        if len(vars_drift_extremo) > 0:
            reporte.append(f"   → CRÍTICO: {len(vars_drift_extremo)} variables con drift extremo")
            reporte.append("   → Considerar eliminarlas si causan problemas")
    else:
        reporte.append("   → No hay drift significativo")
    
    reporte.append("\n3. ESTRATEGIA DE ENTRENAMIENTO:")
    if bajas_agosto > promedio_bajas:
        reporte.append(f"   → Mes {MES_FINAL} tiene más bajas que el promedio")
        reporte.append("   → Considerar aumentar undersampling")
    reporte.append("   → Usar TRAIN_HASTA_FINAL = 202106 (como configurado)")
    reporte.append("   → Esto evita usar datos con drift extremo")
    
    # Guardar reporte
    reporte_text = '\n'.join(reporte)
    
    with open(os.path.join(OUTPUT_DIR, 'REPORTE_FINAL.txt'), 'w', encoding='utf-8') as f:
        f.write(reporte_text)
    
    print(reporte_text)
    print(f"\n{'='*80}")
    print("✓ Reporte guardado: REPORTE_FINAL.txt")
    print(f"{'='*80}")
    
    # Guardar lista consolidada de variables a eliminar
    if len(vars_eliminar) > 0:
        pd.DataFrame({
            'variable': list(vars_eliminar),
            'razon': ['Missing alto o Varianza cero'] * len(vars_eliminar)
        }).to_csv(
            os.path.join(OUTPUT_DIR, 'VARIABLES_A_ELIMINAR.csv'), index=False
        )
        print("✓ Lista de variables a eliminar: VARIABLES_A_ELIMINAR.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Función principal
    """
    print("="*80)
    print("EDA TEMPORAL Y DATA DRIFT ANALYSIS")
    print("="*80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Mes de interés especial: {MES_FINAL}")
    print("="*80)
    
    inicio = datetime.now()
    
    # Crear directorio de salida
    crear_directorio(OUTPUT_DIR)
    
    # Cargar datos
    print("\nCargando dataset...")
    df = pd.read_csv(DATASET_PATH, compression='gzip')
    print(f"✓ Dataset cargado: {df.shape}")
    
    # Calcular clase_ternaria
    df = calcular_clase_ternaria(df)
    
    # ANÁLISIS 1: Registros y clases por mes
    clases_por_mes = analizar_registros_por_mes(df)
    clases_por_mes.to_csv(os.path.join(OUTPUT_DIR, 'clases_por_mes.csv'), index=False)
    
    # ANÁLISIS 2: Missings por mes
    df_missing, vars_problematicas_missing = analizar_missings_por_mes(df)
    
    # ANÁLISIS 3: Data Drift
    df_psi, vars_drift_alto = analizar_data_drift(df)
    
    # ANÁLISIS 4: Evolución de variables clave
    df_evolucion, top_vars = analizar_evolucion_variables(df, n_vars=10)
    df_evolucion.to_csv(os.path.join(OUTPUT_DIR, 'evolucion_variables_clave.csv'), index=False)
    
    # ANÁLISIS 5: Variables constantes
    df_vars, vars_baja_varianza, vars_siempre_constantes = analizar_variables_constantes(df)
    
    # REPORTE FINAL
    generar_reporte_final(
        clases_por_mes, 
        vars_problematicas_missing,
        vars_drift_alto['variable'].tolist() if len(vars_drift_alto) > 0 else [],
        vars_baja_varianza,
        vars_siempre_constantes
    )
    
    # Resumen final
    fin = datetime.now()
    duracion = fin - inicio
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print(f"Duración: {duracion}")
    print(f"Archivos generados en: {OUTPUT_DIR}/")
    print("\nArchivos principales:")
    print("  📊 01_registros_y_clases.png")
    print("  📊 02_evolucion_missings.png")
    print("  📊 03_data_drift_psi.png")
    print("  📊 04_evolucion_variables_clave.png")
    print("  📄 REPORTE_FINAL.txt")
    print("  📄 VARIABLES_A_ELIMINAR.csv")
    print("="*80)


if __name__ == "__main__":
    main()






