import os
import json
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

# Directorios
input_dir = "systemLlama3.3/promptCoNLL"
output_dir = "hola"
bio_dir = "CoNLL"

# Crear directorio de salida si no existe
Path(output_dir).mkdir(parents=True, exist_ok=True)


def process_tagged_text(tagged_text, entities):
    """Convierte el texto etiquetado y entidades en formato BIO."""
    # Reemplazar etiquetas XML por marcadores temporales
    text = tagged_text
    for entity_type in ['LOC', 'ORG', 'PER', 'MISC']:
        text = text.replace(f'<{entity_type}>', f'[{entity_type}_START]')
        text = text.replace(f'</{entity_type}>', f'[{entity_type}_END]')

    # Dividir el texto en tokens
    tokens = text.split()
    bio_tags = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith('[LOC_START]'):
            entity_name = []
            i += 1
            while i < len(tokens) and not tokens[i].endswith('[LOC_END]'):
                entity_name.append(tokens[i])
                i += 1
            if i < len(tokens) and tokens[i].endswith('[LOC_END]'):
                entity_name.append(tokens[i].replace('[LOC_END]', ''))
                i += 1
            if entity_name:
                bio_tags.append((entity_name[0], 'B-LOC'))
                for word in entity_name[1:]:
                    bio_tags.append((word, 'I-LOC'))
        elif token.startswith('[ORG_START]'):
            entity_name = []
            i += 1
            while i < len(tokens) and not tokens[i].endswith('[ORG_END]'):
                entity_name.append(tokens[i])
                i += 1
            if i < len(tokens) and tokens[i].endswith('[ORG_END]'):
                entity_name.append(tokens[i].replace('[ORG_END]', ''))
                i += 1
            if entity_name:
                bio_tags.append((entity_name[0], 'B-ORG'))
                for word in entity_name[1:]:
                    bio_tags.append((word, 'I-ORG'))
        elif token.startswith('[PER_START]'):
            entity_name = []
            i += 1
            while i < len(tokens) and not tokens[i].endswith('[PER_END]'):
                entity_name.append(tokens[i])
                i += 1
            if i < len(tokens) and tokens[i].endswith('[PER_END]'):
                entity_name.append(tokens[i].replace('[PER_END]', ''))
                i += 1
            if entity_name:
                bio_tags.append((entity_name[0], 'B-PER'))
                for word in entity_name[1:]:
                    bio_tags.append((word, 'I-PER'))
        elif token.startswith('[MISC_START]'):
            entity_name = []
            i += 1
            while i < len(tokens) and not tokens[i].endswith('[MISC_END]'):
                entity_name.append(tokens[i])
                i += 1
            if i < len(tokens) and tokens[i].endswith('[MISC_END]'):
                entity_name.append(tokens[i].replace('[MISC_END]', ''))
                i += 1
            if entity_name:
                bio_tags.append((entity_name[0], 'B-MISC'))
                for word in entity_name[1:]:
                    bio_tags.append((word, 'I-MISC'))
        else:
            bio_tags.append((token, 'O'))
            i += 1

    return bio_tags


def save_bio_file(bio_tags, output_path):
    """Guarda las etiquetas BIO en un archivo."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for token, tag in bio_tags:
            f.write(f"{token} {tag}\n")


def read_bio_file(bio_path):
    """Lee un archivo BIO y devuelve tokens y etiquetas."""
    tokens, tags = [], []
    with open(bio_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                token, tag = line.split()
                tokens.append(token)
                tags.append(tag)
    return tokens, tags


def evaluate_bio(pred_tokens, pred_tags, gold_tokens, gold_tags):
    """Evalúa las etiquetas BIO incluso si los tokens no coinciden completamente."""
    # Alinear etiquetas: usar solo las posiciones donde los tokens coinciden
    aligned_pred_tags, aligned_gold_tags = [], []
    min_len = min(len(pred_tokens), len(gold_tokens))

    for i in range(min_len):
        if pred_tokens[i] == gold_tokens[i]:
            aligned_pred_tags.append(pred_tags[i])
            aligned_gold_tags.append(gold_tags[i])
        else:
            # Si los tokens no coinciden, asignar 'O' a ambos para marcar error
            aligned_pred_tags.append('O')
            aligned_gold_tags.append('O')

    # Si hay tokens adicionales en una secuencia, tratarlos como 'O'
    if len(pred_tokens) > min_len:
        aligned_pred_tags.extend(['O'] * (len(pred_tokens) - min_len))
        aligned_gold_tags.extend(['O'] * (len(pred_tokens) - min_len))
    elif len(gold_tokens) > min_len:
        aligned_pred_tags.extend(['O'] * (len(gold_tokens) - min_len))
        aligned_gold_tags.extend(['O'] * (len(gold_tokens) - min_len))

    # Calcular métricas
    precision, recall, f1, _ = precision_recall_fscore_support(aligned_gold_tags, aligned_pred_tags, average='micro')
    return precision, recall, f1, aligned_pred_tags, aligned_gold_tags


# Listas para acumular todas las etiquetas para estadísticas globales
all_pred_tags = []
all_gold_tags = []

# Procesar todos los archivos .txt
for txt_file in Path(input_dir).glob('*.txt'):
    with open(txt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tagged_text = data['tagged_text']
    entities = data['entities']

    # Generar etiquetas BIO
    bio_tags = process_tagged_text(tagged_text, entities)

    # Guardar resultado en la carpeta hola/
    output_path = Path(output_dir) / f"{txt_file.stem}.bio"
    save_bio_file(bio_tags, output_path)
    print(f"Generado: {output_path}")

# Evaluar contra archivos .bio en la carpeta CoNLL
results = []
for pred_bio_file in Path(output_dir).glob('*.bio'):
    # Eliminar el sufijo '_llm_response' del nombre del archivo para buscar la referencia
    gold_bio_filename = pred_bio_file.stem.replace('_llm_response', '') + '.bio'
    gold_bio_file = Path(bio_dir) / gold_bio_filename
    if gold_bio_file.exists():
        pred_tokens, pred_tags = read_bio_file(pred_bio_file)
        gold_tokens, gold_tags = read_bio_file(gold_bio_file)

        # Evaluar incluso si los tokens no coinciden
        metrics = evaluate_bio(pred_tokens, pred_tags, gold_tokens, gold_tags)
        if metrics:
            precision, recall, f1, aligned_pred_tags, aligned_gold_tags = metrics
            results.append({
                'file': pred_bio_file.name,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            # Acumular etiquetas para estadísticas globales
            all_pred_tags.extend(aligned_pred_tags)
            all_gold_tags.extend(aligned_gold_tags)
            print(f"Evaluación para {pred_bio_file.name} (referencia: {gold_bio_filename}):")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
    else:
        print(f"No se encontró archivo de referencia {gold_bio_filename} en {bio_dir}")

# Calcular estadísticas globales
if all_pred_tags and all_gold_tags:
    global_precision, global_recall, global_f1, _ = precision_recall_fscore_support(all_gold_tags, all_pred_tags,
                                                                                    average='micro')
    print("\nEstadísticas globales:")
    print(f"Global Precision: {global_precision:.4f}")
    print(f"Global Recall: {global_recall:.4f}")
    print(f"Global F1: {global_f1:.4f}")
else:
    print("\nNo se pudieron calcular estadísticas globales.")

# Resumen de evaluación por archivo
if results:
    avg_precision = sum(r['precision'] for r in results) / len(results)
    avg_recall = sum(r['recall'] for r in results) / len(results)
    avg_f1 = sum(r['f1'] for r in results) / len(results)
    print("\nResumen de evaluación por archivo:")
    print(f"Promedio Precision: {avg_precision:.4f}")
    print(f"Promedio Recall: {avg_recall:.4f}")
    print(f"Promedio F1: {avg_f1:.4f}")
else:
    print("\nNo se pudieron realizar evaluaciones.")