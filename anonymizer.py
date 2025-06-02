import os
import json
import ast
import re
import xml.etree.ElementTree as ET

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

# Patterns for tagged text - Enhanced to ensure precise extraction
patterns = [
    (r'<([^>]+)>([^<]*)</\1>', lambda m: (m.group(1), m.group(2), m.start(), m.end())),  # <TAG>entidad</TAG>
    (r'<\*([^>]+)>([^<]*)<\*/\1>', lambda m: (m.group(1), m.group(2), m.start(), m.end())),  # <*TAG>entidad</*/TAG>
    (r'<\*([^>*]+)\*>([^<]*)<\*/\1\*>', lambda m: (m.group(1), m.group(2), m.start(), m.end())),
    # <*TAG*>entidad</*/TAG*>
    (r'<<<([^>]+)>>>([^<]*)<<</\1>>>', lambda m: (m.group(1), m.group(2), m.start(), m.end()))
    # <<<TAG>>>entidad<<</TAG>>>
]


def quitar_tildes(text):
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
    }
    result = ''
    for char in text:
        result += replacements.get(char, char)
    return result


def extract_original_text_from_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        text_elem = root.find('TEXT')
        if text_elem is not None:
            return text_elem.text
        else:
            print(f"No TEXT element found in {file_path}")
            return ""
    except Exception as e:
        print(f"Error parsing XML file {file_path}: {e}")
        return ""


def create_xml_from_entities(original_text, entities):
    """
    Create an XML document in MEDDOCAN format from entities dictionary.

    Args:
        original_text (str): The original text without tags
        entities (dict): Dictionary with entity types as keys and lists of entity values as values

    Returns:
        str: XML document as a string
    """
    from collections import Counter

    # Create XML structure
    root = ET.Element("MEDDOCAN")
    text_elem = ET.SubElement(root, "TEXT")
    text_elem.text = original_text
    tags_elem = ET.SubElement(root, "TAGS")

    tag_id = 1

    # Process each entity type and its values
    for tag, values in entities.items():
        # Get the standardized XML tag category
        xml_tag = TAG_CATEGORIES.get(tag)
        if xml_tag is None:
            # Try without accents if the tag is not found
            tag_sin_tildes = quitar_tildes(tag)
            xml_tag = TAG_CATEGORIES.get(tag_sin_tildes, "WARNING")

        # Count occurrences of each value to handle repeated entities
        value_counts = Counter(values)

        for value, count in value_counts.items():
            start_idx = 0
            occurrences = 0

            # Find each occurrence of the entity in the original text
            while occurrences < count:
                start = original_text.find(value, start_idx)
                if start == -1:
                    break  # Value not found or no more occurrences

                end = start + len(value)

                # Create a tag element for this entity occurrence
                ET.SubElement(tags_elem, xml_tag, {
                    "id": str(tag_id),
                    "start": str(start),
                    "end": str(end),
                    "text": value,
                    "TYPE": tag
                })

                tag_id += 1
                start_idx = end
                occurrences += 1

    # Convert the XML tree to a string
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml_content = ET.tostring(root, encoding="unicode", method="xml")

    return xml_declaration + xml_content


def remove_tags_and_track_positions(tagged_text):
    """
    Remove all tags from tagged text and create a mapping between
    original positions and clean text positions.

    Returns:
        clean_text (str): Text without tags
        position_map (list): Maps each position in clean_text to position in tagged_text
    """
    clean_text = ""
    position_map = []

    i = 0
    while i < len(tagged_text):
        char = tagged_text[i]

        # Check if we're at the start of a tag
        if char == '<':
            # Find the end of the tag
            tag_end = tagged_text.find('>', i)
            if tag_end != -1:
                # Skip the entire tag
                i = tag_end + 1
                continue

        # Regular character - add to clean text and track position
        clean_text += char
        position_map.append(i)
        i += 1

    return clean_text, position_map


def find_precise_entity_position(original_text, entity, expected_pos, used_positions):
    """
    Find the exact position of an entity in the original text, avoiding previously used positions.

    Args:
        original_text (str): The original text
        entity (str): The entity to find
        expected_pos (int): Expected position based on clean text alignment
        used_positions (set): Set of positions already used for other entities

    Returns:
        tuple: (start, end) positions or (None, None) if not found
    """
    # Find all possible positions for this entity
    candidates = []
    start = 0

    while True:
        pos = original_text.find(entity, start)
        if pos == -1:
            break

        end_pos = pos + len(entity)

        # Check if this position overlaps with any used position
        overlaps = any(
            (pos < used_end and end_pos > used_start)
            for used_start, used_end in used_positions
        )

        if not overlaps:
            # Calculate distance from expected position
            distance = abs(pos - expected_pos)
            candidates.append((distance, pos, end_pos))

        start = pos + 1

    if candidates:
        # Sort by distance from expected position and return the closest
        candidates.sort()
        return candidates[0][1], candidates[0][2]

    return None, None


def create_xml_from_tagged_text(tagged_text, original_text):
    """
    Create XML from tagged text with improved position tracking.

    Args:
        tagged_text (str): Text with entity tags
        original_text (str): Original text without tags

    Returns:
        str: XML document as a string
    """
    # Create XML structure
    root = ET.Element("MEDDOCAN")
    text_elem = ET.SubElement(root, "TEXT")
    text_elem.text = original_text
    tags_elem = ET.SubElement(root, "TAGS")

    # Remove tags and create position mapping
    clean_text, position_map = remove_tags_and_track_positions(tagged_text)

    # Find all tagged entities
    all_matches = []
    for pattern, extractor in patterns:
        for match in re.finditer(pattern, tagged_text):
            tag, entity, start_pos, end_pos = extractor(match)
            all_matches.append((start_pos, end_pos, tag, entity, match.group(0)))

    # Sort matches by position in tagged text
    all_matches.sort()

    # Track used positions to avoid overlaps
    used_positions = set()
    tag_id = 1

    # Process each match
    for match_start, match_end, tag, entity, full_match in all_matches:
        # Find where the entity content starts in the tagged text
        entity_start_in_tagged = tagged_text.find(entity, match_start)
        if entity_start_in_tagged == -1:
            continue

        # Remove all tags up to this point to find the clean position
        text_before_entity = tagged_text[:entity_start_in_tagged]
        clean_before, _ = remove_tags_and_track_positions(text_before_entity)
        expected_pos_in_original = len(clean_before)

        # Find the precise position in original text
        start_pos, end_pos = find_precise_entity_position(
            original_text, entity, expected_pos_in_original, used_positions
        )

        if start_pos is not None and end_pos is not None:
            # Mark this position as used
            used_positions.add((start_pos, end_pos))

            # Get the standardized XML tag category
            xml_tag = TAG_CATEGORIES.get(tag)
            if xml_tag is None:
                # Try without accents if the tag is not found
                tag_sin_tildes = quitar_tildes(tag)
                xml_tag = TAG_CATEGORIES.get(tag_sin_tildes, "WARNING")

            # Create a tag element for this entity
            ET.SubElement(tags_elem, xml_tag, {
                "id": str(tag_id),
                "start": str(start_pos),
                "end": str(end_pos),
                "text": entity,
                "TYPE": tag
            })

            tag_id += 1
        else:
            print(f"Warning: Could not find precise position for entity '{entity}' with tag '{tag}'")

    # Convert the XML tree to a string
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml_content = ET.tostring(root, encoding="unicode", method="xml")

    return xml_declaration + xml_content


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

            # Extract the tagged text
            tagged_text = data.get('tagged_text', '')

            # Get the base filename without extension
            base_filename = file_name.replace('_llm_response', '')
            xml_path = os.path.join('test/xml', base_filename.replace('.txt', '.xml'))

            # Extract original text from XML file
            original_text = extract_original_text_from_xml(xml_path)

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
    base_input_folders = ['systemLlama3.3', 'systemPhi4']
    base_output_folder = 'procesados_meddocan'

    for base_input in base_input_folders:
        for subfolder in os.listdir(base_input):
            input_path = os.path.join(base_input, subfolder)
            if os.path.isdir(input_path):
                # Modificado para mantener la estructura de directorios
                output_entidades = os.path.join(base_output_folder, base_input, 'entidades', subfolder)
                output_etiquetado = os.path.join(base_output_folder, base_input, 'etiquetado', subfolder)

                print(f"Procesando {input_path}...")
                process_input_files(input_path, output_entidades, output_etiquetado)

    print("Procesamiento completado correctamente")


if __name__ == '__main__':
    main()