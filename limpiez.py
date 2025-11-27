"""
Script de limpieza y monitoreo de memoria para VM con poca RAM
Incluye funciones para liberar memoria y monitorear uso

Autor: Data Scientist Junior
Fecha: 2025-11-26
"""

import gc
import os
import sys
import psutil
import pandas as pd
import numpy as np
import shutil
from datetime import datetime


def obtener_uso_memoria():
    """
    Obtiene el uso actual de memoria del sistema
    """
    memoria = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'ram_total_gb': memoria.total / (1024**3),
        'ram_usada_gb': memoria.used / (1024**3),
        'ram_disponible_gb': memoria.available / (1024**3),
        'ram_porcentaje': memoria.percent,
        'swap_total_gb': swap.total / (1024**3),
        'swap_usada_gb': swap.used / (1024**3),
        'swap_porcentaje': swap.percent
    }


def mostrar_uso_memoria():
    """
    Muestra el uso actual de memoria de forma legible
    """
    uso = obtener_uso_memoria()
    
    print("="*60)
    print("📊 USO DE MEMORIA")
    print("="*60)
    print(f"RAM Total:       {uso['ram_total_gb']:.2f} GB")
    print(f"RAM Usada:       {uso['ram_usada_gb']:.2f} GB ({uso['ram_porcentaje']:.1f}%)")
    print(f"RAM Disponible:  {uso['ram_disponible_gb']:.2f} GB")
    print(f"\nSWAP Total:      {uso['swap_total_gb']:.2f} GB")
    print(f"SWAP Usada:      {uso['swap_usada_gb']:.2f} GB ({uso['swap_porcentaje']:.1f}%)")
    print("="*60)
    
    # Alertas
    if uso['ram_porcentaje'] > 80:
        print("⚠️  ALERTA: RAM por encima del 80%")
    if uso['swap_porcentaje'] > 50:
        print("🔴 CRÍTICO: SWAP siendo usado intensivamente (sistema lento)")
    
    return uso


def limpiar_memoria_agresiva():
    """
    Limpieza agresiva de memoria
    """
    print("\n🧹 Limpiando memoria...")
    
    memoria_antes = obtener_uso_memoria()
    
    # 1. Garbage collection múltiple
    for i in range(3):
        gc.collect()
    
    # 2. Limpiar cache de pandas
    pd.options.mode.copy_on_write = True
    
    memoria_despues = obtener_uso_memoria()
    liberada = memoria_antes['ram_usada_gb'] - memoria_despues['ram_usada_gb']
    
    print(f"✓ Memoria liberada: {liberada:.3f} GB")
    print(f"✓ RAM disponible ahora: {memoria_despues['ram_disponible_gb']:.2f} GB")
    
    return liberada


def limpiar_variables_globales():
    """
    Limpia variables grandes del espacio global
    ¡USAR CON CUIDADO! Solo en scripts, no en notebooks interactivos
    """
    print("\n🗑️  Limpiando variables globales...")
    
    # Obtener variables del módulo principal
    if hasattr(sys.modules['__main__'], '__dict__'):
        main_dict = sys.modules['__main__'].__dict__
        
        variables_grandes = []
        for nombre, obj in list(main_dict.items()):
            if nombre.startswith('_'):
                continue
            
            # Identificar DataFrames grandes
            if isinstance(obj, pd.DataFrame):
                size_mb = obj.memory_usage(deep=True).sum() / (1024**2)
                if size_mb > 10:  # Mayor a 10 MB
                    variables_grandes.append((nombre, 'DataFrame', size_mb))
            
            # Identificar arrays grandes
            elif isinstance(obj, np.ndarray):
                size_mb = obj.nbytes / (1024**2)
                if size_mb > 10:
                    variables_grandes.append((nombre, 'Array', size_mb))
        
        if variables_grandes:
            print("\nVariables grandes encontradas:")
            for nombre, tipo, size_mb in sorted(variables_grandes, key=lambda x: x[2], reverse=True):
                print(f"  {nombre}: {tipo} ({size_mb:.2f} MB)")
            
            respuesta = input("\n¿Eliminar estas variables? (s/n): ")
            if respuesta.lower() == 's':
                for nombre, _, _ in variables_grandes:
                    del main_dict[nombre]
                gc.collect()
                print("✓ Variables eliminadas")
        else:
            print("No se encontraron variables grandes")


def limpiar_directorio_exp(directorio='./exp', dias_antiguos=7, solo_mostrar=True):
    """
    Limpia archivos antiguos del directorio de experimentos
    
    Args:
        directorio: Path al directorio de experimentos
        dias_antiguos: Eliminar archivos con más de N días
        solo_mostrar: Si True, solo muestra lo que borraría sin borrar
    """
    print(f"\n📂 Analizando directorio: {directorio}")
    
    if not os.path.exists(directorio):
        print(f"El directorio {directorio} no existe")
        return
    
    ahora = datetime.now().timestamp()
    limite = dias_antiguos * 24 * 60 * 60  # días a segundos
    
    archivos_viejos = []
    total_size = 0
    
    # Buscar archivos antiguos
    for root, dirs, files in os.walk(directorio):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                mtime = os.path.getmtime(filepath)
                edad = ahora - mtime
                
                if edad > limite:
                    size = os.path.getsize(filepath)
                    archivos_viejos.append((filepath, size, edad / (24*60*60)))
                    total_size += size
            except:
                pass
    
    if archivos_viejos:
        print(f"\n📋 Archivos con más de {dias_antiguos} días:")
        print(f"Total: {len(archivos_viejos)} archivos ({total_size / (1024**2):.2f} MB)")
        
        if solo_mostrar:
            print("\nMuestra de archivos (top 10):")
            for filepath, size, dias in sorted(archivos_viejos, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {filepath}: {size/(1024**2):.2f} MB ({dias:.1f} días)")
            print(f"\nPara eliminarlos, ejecutá:")
            print(f"  limpiar_directorio_exp('{directorio}', dias_antiguos={dias_antiguos}, solo_mostrar=False)")
        else:
            respuesta = input(f"\n¿Eliminar {len(archivos_viejos)} archivos ({total_size / (1024**2):.2f} MB)? (s/n): ")
            if respuesta.lower() == 's':
                for filepath, _, _ in archivos_viejos:
                    try:
                        os.remove(filepath)
                    except:
                        pass
                print(f"✓ {len(archivos_viejos)} archivos eliminados")
                
                # Limpiar directorios vacíos
                for root, dirs, files in os.walk(directorio, topdown=False):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            if not os.listdir(dir_path):
                                os.rmdir(dir_path)
                        except:
                            pass
    else:
        print(f"No hay archivos con más de {dias_antiguos} días")


def optimizar_antes_entrenamiento():
    """
    Función para ejecutar ANTES de entrenar un modelo
    Libera memoria y muestra estado
    """
    print("="*60)
    print("🚀 OPTIMIZANDO SISTEMA ANTES DEL ENTRENAMIENTO")
    print("="*60)
    
    # 1. Mostrar uso actual
    mostrar_uso_memoria()
    
    # 2. Limpiar memoria
    limpiar_memoria_agresiva()
    
    # 3. Mostrar uso después
    print("\n📊 Estado después de la limpieza:")
    uso = mostrar_uso_memoria()
    
    # 4. Recomendaciones
    print("\n💡 RECOMENDACIONES:")
    if uso['ram_porcentaje'] > 70:
        print("  ⚠️  RAM alta: Considerá reducir SEMILLAS_FINAL o usar undersampling más agresivo")
    if uso['swap_porcentaje'] > 10:
        print("  ⚠️  Usando SWAP: El sistema puede estar lento. Considerá reiniciar el kernel")
    if uso['ram_disponible_gb'] < 1:
        print("  🔴 Poca RAM disponible: Muy recomendable reiniciar antes de entrenar")
    else:
        print("  ✓ Sistema en buen estado para entrenar")
    
    print("="*60)


def monitorear_memoria_durante_entrenamiento(intervalo_segundos=60, max_muestras=100):
    """
    Monitorea el uso de memoria durante el entrenamiento
    Útil para detectar memory leaks
    
    USAR EN UN THREAD SEPARADO o llamar manualmente cada N iteraciones
    """
    import time
    
    muestras = []
    
    print("\n📈 Iniciando monitoreo de memoria...")
    print("Presioná Ctrl+C para detener\n")
    
    try:
        for i in range(max_muestras):
            uso = obtener_uso_memoria()
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            muestras.append({
                'timestamp': timestamp,
                'ram_usada_gb': uso['ram_usada_gb'],
                'ram_porcentaje': uso['ram_porcentaje'],
                'swap_usada_gb': uso['swap_usada_gb']
            })
            
            print(f"[{timestamp}] RAM: {uso['ram_usada_gb']:.2f} GB ({uso['ram_porcentaje']:.1f}%) | "
                  f"SWAP: {uso['swap_usada_gb']:.2f} GB")
            
            if i < max_muestras - 1:
                time.sleep(intervalo_segundos)
    
    except KeyboardInterrupt:
        print("\n\n✓ Monitoreo detenido")
    
    # Generar reporte
    if muestras:
        df_muestras = pd.DataFrame(muestras)
        print("\n📊 RESUMEN:")
        print(f"  RAM promedio: {df_muestras['ram_usada_gb'].mean():.2f} GB")
        print(f"  RAM máxima: {df_muestras['ram_usada_gb'].max():.2f} GB")
        print(f"  RAM mínima: {df_muestras['ram_usada_gb'].min():.2f} GB")
        
        # Detectar incremento
        incremento = df_muestras['ram_usada_gb'].iloc[-1] - df_muestras['ram_usada_gb'].iloc[0]
        if incremento > 0.5:
            print(f"\n  ⚠️  Incremento de RAM: +{incremento:.2f} GB (posible memory leak)")
        
        return df_muestras


def verificar_espacio_disco():
    """
    Verifica el espacio disponible en disco
    """
    print("\n💾 ESPACIO EN DISCO:")
    print("="*60)
    
    disco = shutil.disk_usage('.')
    
    total_gb = disco.total / (1024**3)
    usado_gb = disco.used / (1024**3)
    libre_gb = disco.free / (1024**3)
    porcentaje = (usado_gb / total_gb) * 100
    
    print(f"Total:      {total_gb:.2f} GB")
    print(f"Usado:      {usado_gb:.2f} GB ({porcentaje:.1f}%)")
    print(f"Disponible: {libre_gb:.2f} GB")
    
    if libre_gb < 1:
        print("\n🔴 CRÍTICO: Menos de 1 GB disponible")
    elif libre_gb < 5:
        print("\n⚠️  ALERTA: Poco espacio disponible")
    else:
        print("\n✓ Espacio suficiente")
    
    print("="*60)


def diagnostico_completo():
    """
    Realiza un diagnóstico completo del sistema
    """
    print("\n" + "="*60)
    print("🔍 DIAGNÓSTICO COMPLETO DEL SISTEMA")
    print("="*60)
    
    # 1. Memoria
    mostrar_uso_memoria()
    
    # 2. Disco
    verificar_espacio_disco()
    
    # 3. Procesos Python
    print("\n🐍 PROCESOS PYTHON ACTIVOS:")
    print("="*60)
    proceso_actual = psutil.Process()
    memoria_proceso = proceso_actual.memory_info().rss / (1024**2)
    print(f"Proceso actual: {memoria_proceso:.2f} MB")
    
    # 4. Directorio de experimentos
    if os.path.exists('./exp'):
        size = sum(os.path.getsize(os.path.join(root, file))
                   for root, dirs, files in os.walk('./exp')
                   for file in files)
        print(f"\n📂 Directorio ./exp: {size / (1024**2):.2f} MB")
    
    print("="*60)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║           HERRAMIENTAS DE LIMPIEZA DE MEMORIA                  ║
    ╚════════════════════════════════════════════════════════════════╝
    
    FUNCIONES DISPONIBLES:
    
    1. diagnostico_completo()
       → Diagnóstico completo del sistema
    
    2. optimizar_antes_entrenamiento()
       → Ejecutar ANTES de entrenar un modelo
    
    3. limpiar_memoria_agresiva()
       → Libera memoria inmediatamente
    
    4. mostrar_uso_memoria()
       → Muestra el uso actual de RAM y SWAP
    
    5. limpiar_directorio_exp('./exp', dias_antiguos=7)
       → Limpia archivos viejos de experimentos
    
    6. verificar_espacio_disco()
       → Verifica espacio disponible en disco
    
    ═══════════════════════════════════════════════════════════════
    EJECUTANDO DIAGNÓSTICO...
    ═══════════════════════════════════════════════════════════════
    """)
    
    diagnostico_completo()
    
    print("\n" + "="*60)
    print("💡 PARA OPTIMIZAR ANTES DE ENTRENAR:")
    print("   >>> from limpieza_memoria import optimizar_antes_entrenamiento")
    print("   >>> optimizar_antes_entrenamiento()")
    print("="*60)