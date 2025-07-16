import os


def compare_files(pred_dir, gold_dir, output_file="results.txt"):
    # Obtener lista de archivos en ambas carpetas
    pred_files = set(os.listdir(pred_dir))
    gold_files = set(os.listdir(gold_dir))
    common_files = pred_files & gold_files

    metrics = {
        'total_files': len(common_files),
        'failed_files': 0,
        'failed_list': [],
        'precision_sum': 0.0,
        'recall_sum': 0.0,
        'f1_sum': 0.0,
        'processed_files': 0
    }

    for filename in common_files:
        pred_path = os.path.join(pred_dir, filename)
        gold_path = os.path.join(gold_dir, filename)

        try:
            with open(pred_path, 'r', encoding='utf-8') as pf, open(gold_path, 'r', encoding='utf-8') as gf:
                pred_lines = [line.strip() for line in pf if line.strip()]
                gold_lines = [line.strip() for line in gf if line.strip()]

                # Verificar misma cantidad de líneas
                if len(pred_lines) != len(gold_lines):
                    metrics['failed_files'] += 1
                    metrics['failed_list'].append(filename)
                    continue

                # Preparar datos para conlleval
                conll_data = []
                for pred_line, gold_line in zip(pred_lines, gold_lines):
                    pred_parts = pred_line.split()
                    gold_parts = gold_line.split()

                    if not pred_parts or not gold_parts:
                        continue

                    # Tomar la primera palabra y las últimas dos columnas (gold y pred)
                    word = pred_parts[0]
                    gold_tag = gold_parts[-1] if len(gold_parts) > 1 else 'O'
                    pred_tag = pred_parts[-1] if len(pred_parts) > 1 else 'O'

                    conll_data.append(f"{word} {gold_tag} {pred_tag}")

                # Guardar en archivo temporal para conlleval
                temp_file = "temp_conlleval.txt"
                with open(temp_file, 'w', encoding='utf-8') as tf:
                    tf.write("\n".join(conll_data))

                # Ejecutar conlleval.pl y capturar salida
                output = os.popen(f"perl conlleval.pl < {temp_file}").read()

                # Parsear resultados
                precision, recall, f1 = parse_conlleval_output(output)

                if precision is not None and recall is not None and f1 is not None:
                    metrics['precision_sum'] += precision
                    metrics['recall_sum'] += recall
                    metrics['f1_sum'] += f1
                    metrics['processed_files'] += 1

                # Eliminar archivo temporal
                os.remove(temp_file)

        except Exception as e:
            metrics['failed_files'] += 1
            metrics['failed_list'].append(filename)
            print(f"Error processing {filename}: {str(e)}")

    # Calcular promedios
    if metrics['processed_files'] > 0:
        metrics['avg_precision'] = metrics['precision_sum'] / metrics['processed_files']
        metrics['avg_recall'] = metrics['recall_sum'] / metrics['processed_files']
        metrics['avg_f1'] = metrics['f1_sum'] / metrics['processed_files']
    else:
        metrics['avg_precision'] = 0.0
        metrics['avg_recall'] = 0.0
        metrics['avg_f1'] = 0.0

    # Guardar resultados en archivo
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("=== Resultados de la evaluación ===\n")
        out.write(f"Archivos procesados correctamente: {metrics['processed_files']}\n")
        out.write(f"Archivos con errores: {metrics['failed_files']}\n")
        if metrics['failed_files'] > 0:
            out.write("\nArchivos con errores (número de líneas no coincide):\n")
            out.write("\n".join(metrics['failed_list']) + "\n")

        out.write("\nMétricas promedio:\n")
        out.write(f"Precisión: {metrics['avg_precision']:.2f}%\n")
        out.write(f"Recall: {metrics['avg_recall']:.2f}%\n")
        out.write(f"F1-score: {metrics['avg_f1']:.2f}%\n")

    return metrics


def parse_conlleval_output(output):
    """Parse la salida de conlleval.pl para extraer precision, recall y F1"""
    precision = None
    recall = None
    f1 = None

    for line in output.split('\n'):
        if "precision:" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "precision:":
                    precision = float(parts[i + 1].replace('%;', ''))
                elif part == "recall:":
                    recall = float(parts[i + 1].replace('%;', ''))
                elif part == "FB1:":
                    f1 = float(parts[i + 1])

    return precision, recall, f1


if __name__ == "__main__":
    # Configura tus rutas aquí
    PRED_DIR = "hola"  # Carpeta con tus predicciones
    GOLD_DIR = "CoNLL"  # Carpeta con el gold standard
    OUTPUT_FILE = "evaluation_results.txt"

    results = compare_files(PRED_DIR, GOLD_DIR, OUTPUT_FILE)

    print("\n=== Resumen de la evaluación ===")
    print(f"Archivos procesados correctamente: {results['processed_files']}")
    print(f"Archivos con errores: {results['failed_files']}")
    if results['failed_files'] > 0:
        print("\nArchivos con errores:")
        print("\n".join(results['failed_list']))

    print("\nMétricas promedio:")
    print(f"Precisión: {results['avg_precision']:.2f}%")
    print(f"Recall: {results['avg_recall']:.2f}%")
    print(f"F1-score: {results['avg_f1']:.2f}%")

    print(f"\nResultados detallados guardados en {OUTPUT_FILE}")