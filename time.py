import re
from datetime import datetime
import matplotlib.pyplot as plt


def find_processing_period(filepath, target_prompt):
    start_time = None
    end_time = None

    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Buscar el inicio (nuestro prompt específico)
    for i, line in enumerate(lines):
        if f"Procesando con modelo 'llama3.3' y prompt '{target_prompt}'" in line:
            # Extraer timestamp
            match = re.search(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
            if match:
                timestamp_str = match.group(1).replace(',', '.')
                try:
                    start_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                    # Ahora buscar el próximo "Procesando con modelo" después de esta línea
                    for next_line in lines[i + 1:]:
                        if "Procesando con modelo" in next_line:
                            end_match = re.search(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', next_line)
                            if end_match:
                                end_time_str = end_match.group(1).replace(',', '.')
                                end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S.%f')
                            break
                    break
                except ValueError as e:
                    print(f"Error parseando timestamp en línea: {line.strip()}. Error: {e}")

    return start_time, end_time


def calculate_avg_time(start, end, num_operations=235):
    if not start or not end:
        return None
    total_time = (end - start).total_seconds()
    if total_time <= 0:
        print(f"Advertencia: Tiempo total no positivo ({total_time})")
        return None
    return total_time / num_operations


def process_prompts():
    results = {}

    # Procesar prompts 1-10 de logAnonymizer.log
    for i in range(1, 11):
        prompt = f'prompt{i}'
        start, end = find_processing_period('logAnonymizer.log', prompt)

        if not start or not end:
            print(f"No se pudo determinar el período para {prompt} en logAnonymizer.log")
            continue

        avg_time = calculate_avg_time(start, end)

        if avg_time is not None and avg_time > 0:
            results[prompt] = avg_time
        else:
            print(f"Advertencia: Tiempo medio no válido para {prompt}: {avg_time}")

    # Procesar prompt11 de prompt11.log
    start_11, end_11 = find_processing_period('prompt11.log', 'prompt11')
    if start_11 and end_11:
        avg_time_11 = calculate_avg_time(start_11, end_11)
        if avg_time_11 is not None and avg_time_11 > 0:
            results['prompt11'] = avg_time_11
        else:
            print(f"Advertencia: Tiempo medio no válido para prompt11: {avg_time_11}")
    else:
        print("No se pudo determinar el período para prompt11 en prompt11.log")

    return results


def plot_results(results):
    if not results:
        print("No hay datos válidos para graficar")
        return

    prompts = sorted(results.keys(), key=lambda x: int(x[6:]) if x != 'prompt11' else 11)
    avg_times = [results[p] for p in prompts]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(prompts, avg_times, color='skyblue')
    plt.xlabel('Prompt')
    plt.ylabel('Tiempo medio (segundos)')
    plt.title('Tiempo medio de procesamiento por prompt (modelo llama3.3)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Añadir etiquetas con los valores
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.6f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("tiempoMedio.pdf", dpi=300, bbox_inches="tight")
    plt.show()


# Ejecutar el análisis
print("Iniciando análisis de logs...")
results = process_prompts()

if results:
    print("\nResultados obtenidos:")
    for prompt, time in sorted(results.items(), key=lambda x: int(x[0][6:]) if x[0] != 'prompt11' else 11):
        print(f"{prompt}: {time:.6f} segundos")

    plot_results(results)
else:
    print("No se obtuvieron resultados válidos para graficar")