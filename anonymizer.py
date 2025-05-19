import os
import json
import re
import ast

# Category mapping dictionary
TAG_CATEGORIES = {
    # NAME (Nombre)
    'NOMBRE_SUJETO_ASISTENCIA': 'NAME',
    'NOMBRE_PERSONAL_SANITARIO': 'NAME',

    # PROFESSION (Profesión)
    'PROFESION': 'PROFESSION',

    # LOCATION (Ubicación)
    'HOSPITAL': 'LOCATION',
    'INSTITUCION': 'LOCATION',
    'CALLE': 'LOCATION',
    'TERRITORIO': 'LOCATION',
    'PAIS': 'LOCATION',
    'ID_CENTRO_DE_SALUD': 'LOCATION',

    # AGE (Edad)
    'EDAD_SUJETO_ASISTENCIA': 'AGE',

    # DATE (Fechas)
    'FECHAS': 'DATE',

    # CONTACT (Contacto)
    'NUMERO_TELEFONO': 'CONTACT',
    'NUMERO_FAX': 'CONTACT',
    'CORREO_ELECTRONICO': 'CONTACT',
    'URL_WEB': 'CONTACT',

    # ID (Identificadores)
    'ID_ASEGURAMIENTO': 'ID',
    'ID_CONTACTO_ASISTENCIAL': 'ID',
    'NUMERO_BENEF_PLAN_SALUD': 'ID',
    'IDENTIF_VEHICULOS_NRSERIE_PLACAS': 'ID',
    'IDENTIF_DISPOSITIVOS_NRSERIE': 'ID',
    'IDENTIF_BIOMETRICOS': 'ID',
    'ID_SUJETO_ASISTENCIA': 'ID',
    'ID_TITULACION_PERSONAL_SANITARIO': 'ID',
    'ID_EMPLEO_PERSONAL_SANITARIO': 'ID',
    'OTRO_NUMERO_IDENTIF': 'ID',

    # OTHER (Otros)
    'SEXO_SUJETO_ASISTENCIA': 'OTHER',
    'FAMILIARES_SUJETO_ASISTENCIA': 'OTHER',
    'OTROS_SUJETO_ASISTENCIA': 'OTHER',
    'DIREC_PROT_INTERNET': 'OTHER'
}

# Patterns for tagged text
patterns = [
    (r'<([^>]+)>([^<]+)</\1>', lambda m: (m.start(2), m.end(2))),  # <TAG>entidad</TAG>
    (r'<\*([^>]+)>([^<]+)<\*/\1>', lambda m: (m.start(2), m.end(2))),  # <*TAG>entidad</*/TAG>
    (r'<\*([^>*]+)\*>([^<]+)<\*/\1\*>', lambda m: (m.start(2), m.end(2))),  # <*TAG*>entidad</*/TAG*>
    (r'<<<([^>]+)>>>([^<]+)<<</\1>>>', lambda m: (m.start(2), m.end(2)))  # <<<TAG>>>entidad<<</TAG>>>
]


def find_all_positions(text, entity):
    """Find all positions of an entity in the text."""
    entity_positions = []
    start = 0
    while True:
        position = text.find(entity, start)
        if position == -1:
            break
        entity_positions.append((position, position + len(entity)))
        start = position + 1
    return entity_positions


def extract_tagged_entities(tagged_text):
    """Extract entities and their positions from tagged text using regex patterns."""
    tagged_entities = []

    for pattern, position_extractor in patterns:
        matches = re.finditer(pattern, tagged_text)
        for match in matches:
            tag = match.group(1)
            entity_text = match.group(2)
            start, end = position_extractor(match)

            # Track the original positions in the tagged text
            tagged_entities.append({
                'text': entity_text,
                'tag': tag,
                'tagged_start': match.start(),
                'tagged_end': match.end(),
                'entity_start': start,
                'entity_end': end
            })

    return tagged_entities


def create_xml_from_entities(original_text, entities_dict):
    """Create MEDDOCAN XML with one tag per entity occurrence (only first match)."""
    xml = "<?xml version='1.0' encoding='UTF-8'?>\n<MEDDOCAN>\n  <TEXT><![CDATA["
    xml += original_text
    xml += "]]></TEXT>\n  <TAGS>\n"

    tag_id = 1
    used_spans = set()

    for entity_type, entity_values in entities_dict.items():
        category = TAG_CATEGORIES.get(entity_type, 'OTHER')

        for entity_value in entity_values:
            matches = find_all_positions(original_text, entity_value)

            # Use the first unused position
            for start, end in matches:
                if (start, end) not in used_spans:
                    used_spans.add((start, end))
                    xml += f'    <{category} id="T{tag_id}" start="{start}" end="{end}" text="{entity_value}" TYPE="{entity_type}" comment=""/>\n'
                    tag_id += 1
                    break  # Only tag the first available match

    xml += "  </TAGS>\n</MEDDOCAN>"
    return xml


def create_xml_from_tagged_text(tagged_text, original_text):
    """Create MEDDOCAN-style XML from tagged text (with CDATA section)."""
    tagged_entities = extract_tagged_entities(tagged_text)

    used_spans = set()
    tag_id = 1

    xml = "<?xml version='1.0' encoding='UTF-8'?>\n<MEDDOCAN>\n  <TEXT><![CDATA["
    xml += original_text
    xml += "]]></TEXT>\n  <TAGS>\n"

    for entity in tagged_entities:
        entity_text = entity['text']
        tag = entity['tag']

        matches = find_all_positions(original_text, entity_text)

        for start, end in matches:
            if (start, end) not in used_spans:
                used_spans.add((start, end))
                category = TAG_CATEGORIES.get(tag, 'OTHER')
                xml += f'    <{category} id="T{tag_id}" start="{start}" end="{end}" text="{entity_text}" TYPE="{tag}" comment=""/>\n'
                tag_id += 1
                break

    xml += "  </TAGS>\n</MEDDOCAN>"
    return xml


def process_input_files(input_folder, output_entidades_folder, output_etiquetado_folder):
    """Process all input files and generate output XML files."""
    # Create output folders if they don't exist
    os.makedirs(output_entidades_folder, exist_ok=True)
    os.makedirs(output_etiquetado_folder, exist_ok=True)

    # Process each file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_folder, file_name)

            # Read the input file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Extract the JSON data
            try:
                # The content might be a Python dictionary representation
                data = ast.literal_eval(content)
            except (SyntaxError, ValueError):
                # Try parsing as JSON
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    print(f"Error: Could not parse content of {file_name}")
                    continue

            # Extract the original text (remove tags)
            tagged_text = data.get('tagged_text', '')
            original_text = re.sub(r'<[^>]*>|<\*[^>]*>|<\*/[^>]*>|<\*[^>*]*\*>|<\*/[^>*]*\*>|<<<[^>]*>>>|<<</[^>]*>>>',
                                   '', tagged_text)

            # Get entities
            entities = data.get('entities', {})

            # Create XML files
            xml_entidades = create_xml_from_entities(original_text, entities)
            xml_etiquetado = create_xml_from_tagged_text(tagged_text, original_text)

            # Save XML files
            output_file_name = file_name.replace('_llm_response', '').replace('.txt', '.xml')

            with open(os.path.join(output_entidades_folder, output_file_name), 'w', encoding='utf-8') as file:
                file.write(xml_entidades)

            with open(os.path.join(output_etiquetado_folder, output_file_name), 'w', encoding='utf-8') as file:
                file.write(xml_etiquetado)

            print(f"Processed {file_name} successfully")


def main():
    # Define folder paths
    input_folder = 'systemLlama3.3/prompt9'
    output_entidades_folder = 'HOLA/entidades/prompt9'
    output_etiquetado_folder = 'HOLA/etiquetado/prompt9'

    # Process input files
    process_input_files(input_folder, output_entidades_folder, output_etiquetado_folder)
    print("Processing completed successfully")


if __name__ == '__main__':
    main()