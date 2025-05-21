import os
import json
import ast
import re
import unicodedata
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


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
    (r'<([^>]+)>([^<]*)</\1>', lambda m: (m.group(1), m.group(2), m.start(2), m.end(2))),  # <TAG>entidad</TAG>
    (r'<\*([^>]+)>([^<]*)<\*/\1>', lambda m: (m.group(1), m.group(2), m.start(2), m.end(2))),  # <*TAG>entidad</*/TAG>
    (r'<\*([^>*]+)\*>([^<]*)<\*/\1\*>', lambda m: (m.group(1), m.group(2), m.start(2), m.end(2))),
    # <*TAG*>entidad</*/TAG*>
    (r'<<<([^>]+)>>>([^<]*)<<</\1>>>', lambda m: (m.group(1), m.group(2), m.start(2), m.end(2)))
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


def create_xml_from_tagged_text(tagged_text, original_text):
    """
    Create XML from tagged text by matching entity positions in the original text.

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

    # Find all tagged entities in the tagged text
    all_matches = []
    for pattern, extractor in patterns:
        for match in re.finditer(pattern, tagged_text):
            tag, entity, start_pos, end_pos = extractor(match)
            all_matches.append((start_pos, end_pos, tag, entity, match.group(0)))

    # Sort matches by their position in the tagged text
    all_matches.sort()

    # Process the original text and tagged text in parallel
    orig_pos = 0  # Current position in original text
    tagged_pos = 0  # Current position in tagged text
    tag_id = 1

    for _, _, tag, entity, full_match in all_matches:
        # Find where this match starts in the tagged text
        match_start = tagged_text.find(full_match, tagged_pos)
        if match_start == -1:
            continue  # Skip if match not found (shouldn't happen)

        # Calculate how much plain text exists before this tag
        plain_text_before = tagged_text[tagged_pos:match_start]

        # Advance original position by the same amount of plain text
        # This keeps the two texts in sync
        orig_pos += len(plain_text_before)

        # Move the tagged position to after this match
        tagged_pos = match_start + len(full_match)

        # Find the entity in the original text near the expected position
        # Allow for some flexibility in positioning to handle spacing differences
        search_window = 30  # Characters to search around expected position
        search_start = max(0, orig_pos - search_window)
        search_end = min(len(original_text), orig_pos + search_window + len(entity))
        search_region = original_text[search_start:search_end]

        entity_pos = search_region.find(entity)
        if entity_pos != -1:
            # Found the entity within the search window
            start = search_start + entity_pos
            end = start + len(entity)

            # Get the standardized XML tag category
            xml_tag = TAG_CATEGORIES.get(tag)
            if xml_tag is None:
                # Try without accents if the tag is not found
                tag_sin_tildes = quitar_tildes(tag)
                xml_tag = TAG_CATEGORIES.get(tag_sin_tildes, "WARNING")

            # Create a tag element for this entity
            ET.SubElement(tags_elem, xml_tag, {
                "id": str(tag_id),
                "start": str(start),
                "end": str(end),
                "text": entity,
                "TYPE": tag
            })

            # Update the original position to after this entity
            orig_pos = start + len(entity)
            tag_id += 1
        else:
            # If exact match not found, try more aggressive methods
            # For example, ignoring case, removing extra spaces, etc.
            normalized_entity = ' '.join(entity.split())  # Normalize spaces
            normalized_search = ' '.join(search_region.split())

            entity_pos = normalized_search.lower().find(normalized_entity.lower())
            if entity_pos != -1:
                # Need to map back to original position
                start = -1
                current_pos = 0
                normalized_pos = 0

                # Map the normalized position back to the original position
                for i, char in enumerate(search_region):
                    if not char.isspace() or (i > 0 and search_region[i - 1] != ' ' and char == ' '):
                        if normalized_pos == entity_pos:
                            start = search_start + i
                            break
                        normalized_pos += 1

                if start != -1:
                    end = start + len(entity)

                    # Get the standardized XML tag category
                    xml_tag = TAG_CATEGORIES.get(tag)
                    if xml_tag is None:
                        tag_sin_tildes = quitar_tildes(tag)
                        xml_tag = TAG_CATEGORIES.get(tag_sin_tildes, "WARNING")

                    # Create a tag element for this entity
                    ET.SubElement(tags_elem, xml_tag, {
                        "id": str(tag_id),
                        "start": str(start),
                        "end": str(end),
                        "text": entity,
                        "TYPE": tag
                    })

                    tag_id += 1

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