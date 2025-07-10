import os
import urllib.request
import xml.etree.ElementTree as ET
from xml.dom import minidom


def download_conll2002_file(language="es", split="train"):
    """Descarga un archivo CoNLL-2002 desde la URL especificada."""
    base_url = "https://raw.githubusercontent.com/teropa/nlp/master/resources/corpora/conll2002/"
    file_mapping = {
        "es": {
            "train": "esp.train",
            "validation": "esp.testa",
            "test": "esp.testb"
        },
        "nl": {
            "train": "ned.train",
            "validation": "ned.testa",
            "test": "ned.testb"
        }
    }
    file_name = file_mapping[language][split]
    url = f"{base_url}{file_name}"

    # Descargar el archivo
    local_file = f"{file_name}.txt"
    urllib.request.urlretrieve(url, local_file)
    return local_file


def read_conll_file(file_path):
    """Lee un archivo CoNLL-2002 y devuelve una lista de documentos."""
    documents = []
    current_doc = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-") or not line:
                if current_doc:
                    documents.append(current_doc)
                    current_doc = []
                continue
            tokens = line.split()
            if len(tokens) >= 3:  # Token, POS, NER
                current_doc.append(tokens)
    if current_doc:
        documents.append(current_doc)
    return documents


def reconstruct_text(tokens):
    """Reconstruye el texto a partir de los tokens."""
    return " ".join(token[0] for token in tokens)


def calculate_positions(tokens, ner_tags):
    """Calcula las posiciones de inicio y fin de las entidades."""
    entities = []
    current_entity = None
    char_pos = 1
    for i, (token, ner_tag) in enumerate(zip(tokens, ner_tags)):
        token_text = token[0]
        if ner_tag.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'start': char_pos,
                'text': token_text,
                'type': ner_tag[2:]  # PER, LOC, ORG, MISC
            }
        elif ner_tag.startswith('I-') and current_entity:
            current_entity['text'] += f" {token_text}"
        elif current_entity:
            entities.append(current_entity)
            current_entity = None
        char_pos += len(token_text) + 1  # +1 por el espacio
    if current_entity:
        entities.append(current_entity)
    for entity in entities:
        entity['end'] = entity['start'] + len(entity['text'])
    return entities


def map_ner_type(ner_type):
    """Mapea las etiquetas NER de CoNLL-2002 a las de MEDDOCAN."""
    mapping = {
        'PER': 'PER',
        'LOC': 'LOC',
        'ORG': 'ORG',
        'MISC': 'MISC'
    }
    return mapping.get(ner_type, 'OTROS')


def create_xml(documents, output_dir, language, split):
    """Crea archivos XML en el formato de MEDDOCAN."""
    # Crear la carpeta si no existe
    os.makedirs(output_dir, exist_ok=True)

    for doc_id, doc in enumerate(documents):
        root = ET.Element("MEDDOCAN")
        text_elem = ET.SubElement(root, "TEXT")
        text_elem.text = reconstruct_text(doc)
        tags_elem = ET.SubElement(root, "TAGS")

        entities = calculate_positions(doc, [row[2] for row in doc])
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
        # Eliminar la primera línea (la declaración XML que añade minidom)
        xml_str = '\n'.join(xml_str.split('\n')[1:])
        output_file = os.path.join(output_dir, f"{language}_{split}_doc_{doc_id}.xml")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(xml_str)


def create_bio_files(documents, conll_dir, language, split):
    """Crea archivos .bio con las anotaciones BIO en la carpeta CoNLL/"""
    # Crear la carpeta CoNLL si no existe
    os.makedirs(conll_dir, exist_ok=True)

    for doc_id, doc in enumerate(documents):
        bio_file = os.path.join(conll_dir, f"{language}_{split}_doc_{doc_id}.bio")
        with open(bio_file, 'w', encoding='utf-8') as f:
            for token in doc:
                # Formato: TOKEN NER
                f.write(f"{token[0]} {token[2]}\n")
            # Añadir línea en blanco al final para separar documentos
            f.write("\n")


# Configuración
output_dir = "test/conll"  # Directorio donde se guardarán los archivos XML
conll_dir = "CoNLL"       # Carpeta donde se guardarán los archivos BIO (directamente en CoNLL/)
language = "es"           # Cambia a "nl" para holandés
split = "test"            # Cambia a "validation" o "test" si es necesario

# Descargar y procesar el archivo
file_path = download_conll2002_file(language=language, split=split)
documents = read_conll_file(file_path)

# Crear archivos XML
create_xml(documents, output_dir, language, split)

# Crear archivos BIO directamente en CoNLL/
create_bio_files(documents, conll_dir, language, split)

print(f"Archivos XML generados en {output_dir}")
print(f"Archivos BIO generados directamente en {conll_dir}/")

# Opcional: Eliminar el archivo descargado para limpiar
os.remove(file_path)