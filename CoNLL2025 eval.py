import os
import json
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

# Etiquetas válidas
ENTITY_LABELS = {
    'CARDINAL': 'CARDINAL',
    'DATE': 'DATE',
    'EVENT': 'EVENT',
    'FAC': 'FAC',
    'GPE': 'GPE',
    'LANGUAGE': 'LANGUAGE',
    'LAW': 'LAW',
    'LOC': 'LOC',
    'MONEY': 'MONEY',
    'NORP': 'NORP',
    'ORDINAL': 'ORDINAL',
    'ORG': 'ORG',
    'PERCENT': 'PERCENT',
    'PERSON': 'PERSON',
    'PRODUCT': 'PRODUCT',
    'QUANTITY': 'QUANTITY',
    'TIME': 'TIME',
    'WORK_OF_ART': 'WORK_OF_ART',
}

# Directorios
input_dir = "systemLlama3.3/promptCoNLL2025"
output_dir = "hola2025"
bio_dir = "CoNLL2025"

Path(output_dir).mkdir(parents=True, exist_ok=True)


def process_tagged_text(tagged_text, entities):
    """Convierte el texto etiquetado y entidades en formato BIO."""
    text = tagged_text
    for entity_type in ENTITY_LABELS:
        text = text.replace(f'<{entity_type}>', f'[{entity_type}_START]')
        text = text.replace(f'</{entity_type}>', f'[{entity_type}_END]')

    tokens = text.split()
    bio_tags = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        matched = False
        for label in ENTITY_LABELS:
            if token.startswith(f'[{label}_START]'):
                entity_name = []
                i += 1
                while i < len(tokens) and not tokens[i].endswith(f'[{label}_END]'):
                    entity_name.append(tokens[i])
                    i += 1
                if i < len(tokens) and tokens[i].endswith(f'[{label}_END]'):
                    entity_name.append(tokens[i].replace(f'[{label}_END]', ''))
                    i += 1
                if entity_name:
                    bio_tags.append((entity_name[0], f'B-{label}'))
                    for word in entity_name[1:]:
                        bio_tags.append((word, f'I-{label}'))
                matched = True
                break
        if not matched:
            bio_tags.append((token, 'O'))
            i += 1

    return bio_tags


def save_bio_file(bio_tags, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for token, tag in bio_tags:
            f.write(f"{token} {tag}\n")


def read_bio_file(bio_path):
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
    aligned_pred_tags, aligned_gold_tags = [], []
    min_len = min(len(pred_tokens), len(gold_tokens))

    for i in range(min_len):
        if pred_tokens[i] == gold_tokens[i]:
            aligned_pred_tags.append(pred_tags[i])
            aligned_gold_tags.append(gold_tags[i])
        else:
            aligned_pred_tags.append('O')
            aligned_gold_tags.append('O')

    if len(pred_tokens) > min_len:
        aligned_pred_tags.extend(['O'] * (len(pred_tokens) - min_len))
        aligned_gold_tags.extend(['O'] * (len(pred_tokens) - min_len))
    elif len(gold_tokens) > min_len:
        aligned_pred_tags.extend(['O'] * (len(gold_tokens) - min_len))
        aligned_gold_tags.extend(['O'] * (len(gold_tokens) - min_len))

    precision, recall, f1, _ = precision_recall_fscore_support(aligned_gold_tags, aligned_pred_tags, average='micro')
    return precision, recall, f1, aligned_pred_tags, aligned_gold_tags


all_pred_tags = []
all_gold_tags = []

for txt_file in Path(input_dir).glob('*.txt'):
    with open(txt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tagged_text = data['tagged_text']
    entities = data['entities']
    bio_tags = process_tagged_text(tagged_text, entities)

    output_path = Path(output_dir) / f"{txt_file.stem}.bio"
    save_bio_file(bio_tags, output_path)
    print(f"Generado: {output_path}")

results = []
for pred_bio_file in Path(output_dir).glob('*.bio'):
    gold_bio_filename = pred_bio_file.stem.replace('_llm_response', '') + '.bio'
    gold_bio_file = Path(bio_dir) / gold_bio_filename
    if gold_bio_file.exists():
        pred_tokens, pred_tags = read_bio_file(pred_bio_file)
        gold_tokens, gold_tags = read_bio_file(gold_bio_file)

        metrics = evaluate_bio(pred_tokens, pred_tags, gold_tokens, gold_tags)
        if metrics:
            precision, recall, f1, aligned_pred_tags, aligned_gold_tags = metrics
            results.append({
                'file': pred_bio_file.name,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            all_pred_tags.extend(aligned_pred_tags)
            all_gold_tags.extend(aligned_gold_tags)
            print(f"Evaluación para {pred_bio_file.name} (referencia: {gold_bio_filename}):")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
    else:
        print(f"No se encontró archivo de referencia {gold_bio_filename} en {bio_dir}")

if all_pred_tags and all_gold_tags:
    global_precision, global_recall, global_f1, _ = precision_recall_fscore_support(
        all_gold_tags, all_pred_tags, average='micro'
    )
    print("\nEstadísticas globales:")
    print(f"Global Precision: {global_precision:.4f}")
    print(f"Global Recall: {global_recall:.4f}")
    print(f"Global F1: {global_f1:.4f}")
else:
    print("\nNo se pudieron calcular estadísticas globales.")

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
