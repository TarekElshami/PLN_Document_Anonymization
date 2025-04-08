import requests
import xml.etree.ElementTree as ET
import os
import re
import unicodedata


def extract_entities(text):
    """
    Usa el modelo LLM en el servidor para identificar entidades nombrables en el texto.
    """
    url = "http://localhost:20201/api/generate"
    payload = {
        "model": "llama3.3",
        "prompt": f"""
        Lee el siguiente texto y extrae únicamente el nombre del paciente, sus apellidos y el nombre completo del médico, **sin incluir información adicional ni campos extra**.

        Reglas estrictas:
        1. NOMBRE_PACIENTE: Extrae solo el nombre o nombres del paciente, sin apellidos ni caracteres adicionales (puntos, comas, etc.).
        2. APELLIDOS_PACIENTE: Extrae solo los apellidos del paciente, sin nombres ni caracteres adicionales.
        3. MEDICO: Extrae el nombre y los apellidos completos del médico, sin títulos ("Dr.", "Dra."), servicios médicos, números de colegiado, ni caracteres adicionales.

        Formato de salida (exacto, sin cambios ni caracteres adicionales):
        NOMBRE_PACIENTE: [Nombre o nombres del paciente]  
        APELLIDOS_PACIENTE: [Apellidos del paciente únicamente]  
        MEDICO: [Nombre y apellidos del médico únicamente]  

        IMPORTANTE:
        - NO devuelvas texto adicional (títulos, especialidades, hospitales, números de colegiado, etc.).
        - NO generes campos extra como "NOMBRE_COMPLETO_MEDICO".
        - NO agregues puntos, comas ni espacios innecesarios al final de los nombres o apellidos.
        - Si no puedes identificar alguna entidad, deja el campo vacío (ejemplo: "NOMBRE_PACIENTE: ").

        Ejemplos:
        Ejemplo 1:
        Texto: "El paciente Juan Hidalgo Manzana fue atendido por el Dr. Tarek Elshami Ahmed del Servicio de Urología, NºCol: 46 28 52938."
        Salida esperada:  
        NOMBRE_PACIENTE: Juan  
        APELLIDOS_PACIENTE: Hidalgo Manzana  
        MEDICO: Tarek Elshami Ahmed  

        Ejemplo 2:
        Texto: "José Luis Redondo de Oro acudió a consulta con el doctor Luis Enrique de la Fuente."
        Salida esperada:  
        NOMBRE_PACIENTE: José Luis  
        APELLIDOS_PACIENTE: Redondo de Oro  
        MEDICO: Luis Enrique de la Fuente  

        Ejemplo 3:
        Texto: "Paciente sin nombre identificado, atendido por Dra. Paula de Arriba Crespo."
        Salida esperada:  
        NOMBRE_PACIENTE:  
        APELLIDOS_PACIENTE:  
        MEDICO: Paula de Arriba Crespo

        El texto que debes analizar es el siguiente: {text}
        """,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=1000)
        response.raise_for_status()  # Lanza excepción si hay error HTTP
        print("Respuesta cruda del modelo: \n", response.json()['response'])  # Depuración
        return parse_llm_response(response.json()['response'])
    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con el servidor: {e}")
        return {'NOMBRE_PACIENTE': '', 'APELLIDOS_PACIENTE': '', 'MEDICO': ''}


def parse_llm_response(response):
    """
    Parsea la respuesta del modelo LLM para extraer las entidades nombradas.
    """
    entities = {
        'NOMBRE_PACIENTE': '',
        'APELLIDOS_PACIENTE': '',
        'MEDICO': ''
    }

    if not response or not isinstance(response, str):
        print("Respuesta inválida o vacía del modelo")
        return entities

    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('NOMBRE_PACIENTE:'):
            entities['NOMBRE_PACIENTE'] = line.replace('NOMBRE_PACIENTE:', '').strip()
        elif line.startswith('APELLIDOS_PACIENTE:'):
            entities['APELLIDOS_PACIENTE'] = line.replace('APELLIDOS_PACIENTE:', '').strip()
        elif line.startswith('MEDICO:'):
            entities['MEDICO'] = line.replace('MEDICO:', '').strip()

    return entities


def normalize_text(text):
    """
    Normaliza el texto eliminando tildes y otros diacríticos.
    """
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def find_all_matches(text, entities):
    """
    Encuentra todas las apariciones de cada entidad en el texto, considerando versiones con y sin tildes.
    """
    matches = {
        'NOMBRE_PACIENTE': [],
        'APELLIDOS_PACIENTE': [],
        'MEDICO': []
    }

    normalized_text = normalize_text(text)  # Texto sin tildes

    for entity, value in entities.items():
        if value:
            normalized_value = normalize_text(value)  # Entidad sin tildes

            # Buscar todas las apariciones (sin tildes) en el texto normalizado
            for match in re.finditer(re.escape(normalized_value), normalized_text, re.IGNORECASE):
                start_norm, end_norm = match.span()

                # Recuperar la versión exacta del texto original
                start, end = start_norm, end_norm
                original_match = text[start:end]  # Extraer texto exacto del documento

                matches[entity].append({
                    'text': original_match,  # Preserva el texto original con sus tildes
                    'start': start,
                    'end': end
                })

    return matches


def resolve_overlaps(matches):
    """
    Resuelve solapamientos entre entidades y prioriza la coincidencia más larga.
    """
    # Ordenar todas las coincidencias por longitud (de mayor a menor)
    all_matches = []
    for entity, entity_matches in matches.items():
        for match in entity_matches:
            all_matches.append({
                'entity': entity,
                'text': match['text'],
                'start': match['start'],
                'end': match['end'],
                'length': match['end'] - match['start']
            })

    # Ordenar por longitud (priorizar las coincidencias más largas)
    all_matches.sort(key=lambda x: x['length'], reverse=True)

    # Seleccionar las coincidencias sin solapamientos
    selected_matches = []
    used_positions = set()

    for match in all_matches:
        overlap = False
        for pos in range(match['start'], match['end']):
            if pos in used_positions:
                overlap = True
                break

        if not overlap:
            selected_matches.append(match)
            for pos in range(match['start'], match['end']):
                used_positions.add(pos)

    return selected_matches


def generate_xml_output(text, entities):
    """
    Genera el XML con todas las apariciones de las entidades nombradas.
    """
    # Crear el elemento raíz
    meddocan = ET.Element('MEDDOCAN')

    # Crear el elemento TEXT
    text_element = ET.SubElement(meddocan, 'TEXT')
    text_element.text = f"<![CDATA[{text}]]>"

    # Crear el elemento TAGS
    tags_element = ET.SubElement(meddocan, 'TAGS')

    # Encontrar todas las apariciones de las entidades
    matches = find_all_matches(text, entities)

    # Resolver solapamientos y priorizar coincidencias
    resolved_matches = resolve_overlaps(matches)

    # Asignar identificadores únicos
    id_counter = 1

    for match in resolved_matches:
        if match['entity'] == 'NOMBRE_PACIENTE':
            name_element = ET.SubElement(tags_element, 'NAME')
            name_element.set('id', f'T{id_counter}')
            name_element.set('TYPE', 'NOMBRE_SUJETO_ASISTENCIA')
            id_counter += 1
            name_element.set('start', str(match['start']))
            name_element.set('end', str(match['end']))
            name_element.set('text', match['text'])
            name_element.set('comment', '')

        elif match['entity'] == 'APELLIDOS_PACIENTE':
            name_element = ET.SubElement(tags_element, 'NAME')
            name_element.set('id', f'T{id_counter}')
            name_element.set('TYPE', 'NOMBRE_SUJETO_ASISTENCIA')
            id_counter += 1
            name_element.set('start', str(match['start']))
            name_element.set('end', str(match['end']))
            name_element.set('text', match['text'])
            name_element.set('comment', '')

        elif match['entity'] == 'MEDICO':
            name_element = ET.SubElement(tags_element, 'NAME')
            name_element.set('id', f'T{id_counter}')
            name_element.set('TYPE', 'NOMBRE_PERSONAL_SANITARIO')
            id_counter += 1
            name_element.set('start', str(match['start']))
            name_element.set('end', str(match['end']))
            name_element.set('text', match['text'])
            name_element.set('comment', '')

    xml_str = ET.tostring(meddocan, encoding='utf-8').decode('utf-8')
    return xml_str


def post_process_output(output):
    """
    Post-procesa la salida para eliminar caracteres no deseados y asegurar el formato exacto, preservando puntos de abreviaturas.
    """
    if not output or output == 'None':
        return ''
    # Eliminar títulos, "NCol", y texto adicional, pero no puntos dentro del nombre
    cleaned_output = re.sub(r'(Dr\.?|Dra\.?|NCol|NºCol:.*|Servicio.*|\d+.*|\s+$)', '', output).strip()
    # Eliminar solo el punto final si existe, preservando puntos internos
    cleaned_output = re.sub(r'\.$', '', cleaned_output)
    # Eliminar espacios al final de la cadena
    cleaned_output = cleaned_output.rstrip()
    return cleaned_output


def process_xml_files(input_dir, output_dir):
    """
    Procesa los archivos XML usando el modelo LLM.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    total_files = len(xml_files)

    for i, filename in enumerate(xml_files, start=1):
        if filename.endswith('.xml'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                print(f"Procesando archivo {i} de {total_files}: {filename}")
                # Leer el XML original
                tree = ET.parse(input_path)
                root = tree.getroot()

                # Obtener el texto
                text_element = root.find('.//TEXT')
                if text_element is None:
                    raise ValueError(f"No se encontró el elemento TEXT en el archivo {filename}")

                text = text_element.text if text_element.text else ""

                # Extraer entidades usando el modelo LLM
                entities = extract_entities(text)

                # Post-procesar cada entidad individualmente
                entities = {k: post_process_output(v) for k, v in entities.items()}

                # Imprimir resultado filtrado
                print(f"NOMBRE_PACIENTE: {entities['NOMBRE_PACIENTE']}")
                print(f"APELLIDOS_PACIENTE: {entities['APELLIDOS_PACIENTE']}")
                print(f"MEDICO: {entities['MEDICO']}")

                # Generar nuevo XML
                output_xml = generate_xml_output(text, entities)

                # Guardar el resultado
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                    f.write(output_xml)

                print(f"Procesado y guardado: {output_path}")
                print("*" * 80)

            except ET.ParseError as e:
                print(f"Error de parseo en {filename}: {e}")
            except ValueError as e:
                print(f"Error en {filename}: {e}")
            except Exception as e:
                print(f"Error inesperado en {filename}: {e}")


if __name__ == "__main__":
    input_directory = 'test/xml'
    output_directory = 'output/xml/best_LLaMA_model_nameTAG'
    process_xml_files(input_directory, output_directory)