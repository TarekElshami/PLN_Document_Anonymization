import re
import json
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def normalizar_prompt(prompt_path):
    """Extrae solo el nombre del prompt final del path, por ejemplo 'prompt1'."""
    return prompt_path.strip().split("/")[-1]

def ordenar_prompts_numericamente(columnas):
    """Ordena los prompts por número (prompt1, prompt2, ..., prompt10)."""
    def extract_num(prompt):
        match = re.search(r'\d+', prompt)
        return int(match.group()) if match else float('inf')
    return sorted(columnas, key=extract_num)

def limpiar_y_parsear_json(json_string):
    """
    Intenta limpiar y parsear una cadena JSON.
    Retorna True si el parseo es exitoso y el tipo de limpieza, False y None en caso contrario.
    Tipos de limpieza: 'delimiters', 'no_delimiters', 'raw'.
    """
    # 1. Intentar con delimitadores ```json
    json_block_match = re.search(r'```json\s*(.*?)\s*```', json_string, re.DOTALL)
    if json_block_match:
        clean_json_string = json_block_match.group(1)
        try:
            json.loads(clean_json_string)
            return True, 'delimiters'
        except json.JSONDecodeError:
            pass # Falló con delimitadores, probar otras opciones

    # 2. Intentar con delimitadores de comillas (si no hay ```json)
    if not json_block_match: # Solo si no se encontró el bloque ```json
        json_quotes_match = re.search(r'```(?!"json)\s*(.*?)\s*```', json_string, re.DOTALL)
        if json_quotes_match:
            clean_json_string = json_quotes_match.group(1)
            try:
                json.loads(clean_json_string)
                return True, 'no_delimiters'
            except json.JSONDecodeError:
                pass # Falló con comillas, probar raw

    # 3. Intentar parsear la cadena tal cual (asumiendo que es JSON puro)
    try:
        json.loads(json_string.strip())
        return True, 'raw'
    except json.JSONDecodeError:
        return False, None


def analizar_log_y_obtener_stats(path_log):
    """
    Analiza el archivo de log para extraer estadísticas de fallos de JSON,
    clasificando los errores y registrando los archivos fallidos.
    """
    if not os.path.exists(path_log):
        print(f"Error: El archivo de log no se encuentra en la ruta especificada: {path_log}")
        return None, None, None, None, None, None
    with open(path_log, "r", encoding="utf-8") as f:
        log = f.read()

    bloques = re.split(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - Procesando archivo:", log)
    if bloques and bloques[0].strip() == "":
        bloques = bloques[1:]

    total_por_modelo_y_prompt = defaultdict(lambda: defaultdict(int))
    errores_json_total = defaultdict(lambda: defaultdict(int))
    errores_delimitadores = defaultdict(lambda: defaultdict(int)) # Fallo por ```json
    errores_sin_json = defaultdict(lambda: defaultdict(int))      # Fallo por ``` (sin json)
    errores_json_malformado = defaultdict(lambda: defaultdict(int)) # JSON interno malformado

    # Para registrar los archivos que fallaron *solo* por JSON malformado
    archivos_fallidos_json_malformado = defaultdict(lambda: defaultdict(list))
    todos_los_archivos_fallidos_json_malformado = []

    for bloque in bloques:
        modelo_match = re.search(r"Modelo:\s(\S+)", bloque)
        modelo = modelo_match.group(1) if modelo_match else None
        if not modelo or modelo == "desconocido":
            continue

        prompt_match = re.search(r"Directorio de salida:\s(\S+)", bloque)
        prompt = normalizar_prompt(prompt_match.group(1)) if prompt_match else "prompt_desconocido"

        archivo_actual_match = re.search(r"^(.*?\.xml)", bloque, re.MULTILINE)
        archivo_actual = archivo_actual_match.group(1).strip() if archivo_actual_match else "asffd"

        respuesta_cruda_match = re.search(r"Respuesta cruda del modelo:\s*\n(.*?)(?=\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - Resumen final de entidades encontradas:|\Z)", bloque, re.DOTALL)
        respuesta_cruda = respuesta_cruda_match.group(1).strip() if respuesta_cruda_match else ""

        total_por_modelo_y_prompt[modelo][prompt] += 1

        if "Error parseando JSON" in bloque:
            errores_json_total[modelo][prompt] += 1

            es_valido, tipo_limpieza = limpiar_y_parsear_json(respuesta_cruda)

            if not es_valido:
                errores_json_malformado[modelo][prompt] += 1
                # Solo añade a la lista de fallidos si el JSON está malformado
                archivos_fallidos_json_malformado[modelo][prompt].append(archivo_actual)
                todos_los_archivos_fallidos_json_malformado.append(archivo_actual)
            elif tipo_limpieza == 'delimiters':
                errores_delimitadores[modelo][prompt] += 1
            elif tipo_limpieza == 'no_delimiters':
                errores_sin_json[modelo][prompt] += 1

    # Crear DataFrames para tasas de fallo
    data_tasa_fallo = []
    for modelo in total_por_modelo_y_prompt:
        for prompt in total_por_modelo_y_prompt[modelo]:
            total = total_por_modelo_y_prompt[modelo][prompt]
            errores = errores_json_total[modelo][prompt]
            tasa_fallo = errores / total if total else 0
            data_tasa_fallo.append((modelo, prompt, tasa_fallo))

    df_tasa_fallo = pd.DataFrame(data_tasa_fallo, columns=["Modelo", "Prompt", "Tasa de fallo"])
    pivot_df_tasa_fallo = df_tasa_fallo.pivot(index="Modelo", columns="Prompt", values="Tasa de fallo")
    ordered_columns = ordenar_prompts_numericamente(pivot_df_tasa_fallo.columns)
    pivot_df_tasa_fallo = pivot_df_tasa_fallo[ordered_columns]

    # Crear DataFrames para conteos específicos
    data_counts = []
    for modelo in total_por_modelo_y_prompt:
        for prompt in total_por_modelo_y_prompt[modelo]:
            total = total_por_modelo_y_prompt[modelo][prompt]
            errores_total_val = errores_json_total[modelo][prompt]
            errores_delim_val = errores_delimitadores[modelo][prompt]
            errores_sin_json_val = errores_sin_json[modelo][prompt]
            errores_malf_val = errores_json_malformado[modelo][prompt]
            data_counts.append((modelo, prompt, total, errores_total_val, errores_delim_val, errores_sin_json_val, errores_malf_val))

    df_counts = pd.DataFrame(data_counts, columns=["Modelo", "Prompt", "Total Peticiones", "Errores JSON (Total)", "Errores por Delimitadores (```json)", "Errores por Delimitadores (``` sin json)", "Errores por JSON Malformado"])

    pivot_df_total_peticiones = df_counts.pivot(index="Modelo", columns="Prompt", values="Total Peticiones")
    pivot_df_total_peticiones = pivot_df_total_peticiones[ordered_columns]

    pivot_df_errores_total = df_counts.pivot(index="Modelo", columns="Prompt", values="Errores JSON (Total)")
    pivot_df_errores_total = pivot_df_errores_total[ordered_columns]

    pivot_df_errores_delimitadores = df_counts.pivot(index="Modelo", columns="Prompt", values="Errores por Delimitadores (```json)")
    pivot_df_errores_delimitadores = pivot_df_errores_delimitadores[ordered_columns]

    pivot_df_errores_sin_json = df_counts.pivot(index="Modelo", columns="Prompt", values="Errores por Delimitadores (``` sin json)")
    pivot_df_errores_sin_json = pivot_df_errores_sin_json[ordered_columns]

    pivot_df_errores_json_malformado = df_counts.pivot(index="Modelo", columns="Prompt", values="Errores por JSON Malformado")
    pivot_df_errores_json_malformado = pivot_df_errores_json_malformado[ordered_columns]

    # Identificar archivos comunes
    common_failed_files = {}
    if len(todos_los_archivos_fallidos_json_malformado) > 1:
        from collections import Counter
        file_counts = Counter(todos_los_archivos_fallidos_json_malformado)
        common_failed_files = {file: count for file, count in file_counts.items() if count > 1}

    return pivot_df_tasa_fallo, pivot_df_total_peticiones, pivot_df_errores_total, pivot_df_errores_delimitadores, pivot_df_errores_sin_json, pivot_df_errores_json_malformado, archivos_fallidos_json_malformado, common_failed_files

def generar_heatmap(df_heatmap, title, cbar_label, fmt=".0%"):
    """Genera y muestra un heatmap."""
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    ax = sns.heatmap(
        df_heatmap,
        cmap="rocket_r",
        linewidths=0.5,
        linecolor='gray',
        cbar=True,
        square=True,
        annot=True,
        fmt=fmt,
        annot_kws={"fontsize": 9}
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_label, fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return plt.gcf()
