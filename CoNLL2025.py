import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pandas as pd


def download_conll2025_dataset():
    """Descarga el dataset CONLL2025 desde Hugging Face."""
    try:
        df = pd.read_parquet("hf://datasets/boltuix/conll2025-ner/conll2025_ner.parquet")
        return df
    except Exception as e:
        print(f"Error al descargar el dataset: {str(e)}")
        return None


def process_conll2025_data(df, split="test"):
    """Procesa los datos de CONLL2025 y devuelve una lista de documentos."""
    documents = []

    # Verificar que las columnas existen
    if 'tokens' not in df.columns or 'ner_tags' not in df.columns:
        print("Error: Las columnas 'tokens' o 'ner_tags' no existen en el dataset")
        return documents

    # Filtrar por split
    split_data = df[df["split"] == split]

    for _, row in split_data.iterrows():
        tokens = row["tokens"]
        ner_tags = row["ner_tags"]

        # Combinar tokens con sus etiquetas NER (similar al formato CoNLL)
        doc = [[token, '_', tag] for token, tag in zip(tokens, ner_tags)]
        documents.append(doc)

    return documents


def reconstruct_text(tokens):
    """Reconstruye el texto a partir de los tokens."""
    return " ".join(token[0] for token in tokens)


def calculate_positions(tokens):
    """Calcula las posiciones de inicio y fin de las entidades basándose en el texto reconstruido."""
    entities = []
    current_entity = None

    # Reconstruir el texto completo para calcular posiciones precisas
    full_text = reconstruct_text(tokens)
    char_pos = 0

    for i, token_data in enumerate(tokens):
        token_text = token_data[0]
        ner_tag = token_data[2]

        # Encontrar la posición del token en el texto completo
        token_start = full_text.find(token_text, char_pos)
        if token_start == -1:
            # Si no se encuentra, usar la posición estimada
            token_start = char_pos

        if ner_tag.startswith('B-'):
            # Finalizar entidad anterior si existe
            if current_entity:
                entities.append(current_entity)

            # Iniciar nueva entidad
            current_entity = {
                'start': token_start,
                'end': token_start + len(token_text),
                'text': token_text,
                'type': ner_tag[2:]  # PER, LOC, ORG, etc.
            }
        elif ner_tag.startswith('I-') and current_entity:
            # Extender entidad actual
            current_entity['end'] = token_start + len(token_text)
            # Reconstruir el texto de la entidad desde el texto completo
            current_entity['text'] = full_text[current_entity['start']:current_entity['end']]
        elif current_entity:
            # Finalizar entidad actual
            entities.append(current_entity)
            current_entity = None

        # Actualizar posición para el siguiente token
        char_pos = token_start + len(token_text) + 1  # +1 por el espacio

    # Agregar la última entidad si existe
    if current_entity:
        entities.append(current_entity)

    return entities


def map_ner_type(ner_type):
    """Mapea las etiquetas NER a un formato estándar."""
    # Puedes ajustar este mapeo según las etiquetas específicas de CONLL2025
    return ner_type if ner_type in ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART'] else 'OTROS'


def create_xml(documents, output_dir, language, split):
    """Crea archivos XML en el formato de MEDDOCAN."""
    os.makedirs(output_dir, exist_ok=True)

    for doc_id, doc in enumerate(documents):
        root = ET.Element("MEDDOCAN")
        text_elem = ET.SubElement(root, "TEXT")
        text_elem.text = reconstruct_text(doc)
        tags_elem = ET.SubElement(root, "TAGS")

        entities = calculate_positions(doc)
        for i, entity in enumerate(entities):
            tag_elem = ET.SubElement(tags_elem, entity['type'])
            tag_elem.set('id', f"T{i + 1}")
            tag_elem.set('start', str(entity['start']))
            tag_elem.set('end', str(entity['end']))
            tag_elem.set('text', entity['text'])
            tag_elem.set('TYPE', map_ner_type(entity['type']))
            tag_elem.set('comment', '')

        # Guardar el archivo XML
        xml_str = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml(indent="  ")
        xml_str = '\n'.join(xml_str.split('\n')[1:])
        output_file = os.path.join(output_dir, f"{language}_{split}_doc_{doc_id}.xml")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(xml_str)


def create_bio_files(documents, conll_dir, language, split):
    """Crea archivos .bio con las anotaciones BIO en la carpeta CoNLL/"""
    os.makedirs(conll_dir, exist_ok=True)

    for doc_id, doc in enumerate(documents):
        bio_file = os.path.join(conll_dir, f"{language}_{split}_doc_{doc_id}.bio")
        with open(bio_file, 'w', encoding='utf-8') as f:
            for token in doc:
                f.write(f"{token[0]} {token[2]}\n")
            f.write("\n")


# Configuración
output_dir = "test/conll2025"  # Directorio para archivos XML
conll_dir = "CoNLL2025"  # Carpeta para archivos BIO
language = "en"  # Idioma (ajustar según dataset)
split = "test"  # Split a procesar

# Descargar y procesar el dataset
df = download_conll2025_dataset()
if df is not None:
    documents = process_conll2025_data(df, split=split)

    if documents:
        # Crear archivos XML
        create_xml(documents, output_dir, language, split)

        # Crear archivos BIO
        create_bio_files(documents, conll_dir, language, split)

        print(f"Archivos XML generados en {output_dir}")
        print(f"Archivos BIO generados directamente en {conll_dir}/")
        print(f"Procesados {len(documents)} documentos")
    else:
        print("No se encontraron documentos para procesar")
else:
    print("No se pudo procesar el dataset. Verifica los errores anteriores.")