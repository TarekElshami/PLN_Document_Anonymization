import requests
import xml.etree.ElementTree as ET
import os

def extract_entities(text):
    """
    Usa el modelo LLM en el servidor para identificar entidades nombrables en el texto.
    """
    url = "http://localhost:20201/api/generate"
    payload = {
        "model": "llama3.2:1b",
        "prompt": f"""
        Lee el siguiente texto y extrae únicamente el nombre del paciente, sus apellidos y el nombre completo del médico, **sin incluir información adicional ni campos extra**.

        ### Reglas estrictas:
        1. **NOMBRE_PACIENTE**: Extrae solo el nombre o nombres del paciente, sin apellidos.
        2. **APELLIDOS_PACIENTE**: Extrae solo los apellidos del paciente, sin incluir nombres.
        3. **MEDICO**: Extrae el nombre y los apellidos completos del médico, sin:
           - Títulos como "Dr.", "Dra.", "Médico".
           - Servicios médicos ("Servicio de Urología", "Unidad de Cirugía").
           - Números de colegiado ("NºCol: 46 28 52938").
           - Campos adicionales como "NOMBRE_COMPLETO_MEDICO".

        ### Formato de salida (exacto, sin cambios ni caracteres adicionales):
        NOMBRE_PACIENTE: [Nombre o nombres del paciente]  
        APELLIDOS_PACIENTE: [Apellidos del paciente únicamente]  
        MEDICO: [Nombre y apellidos del médico únicamente]  

        **IMPORTANTE**
        - **NO devuelvas texto adicional** (títulos, especialidades, hospitales, números de colegiado).  
        - **NO generes campos extra** como "NOMBRE_COMPLETO_MEDICO".  
        - **No agregues puntos ni caracteres al final de los apellidos o nombres.**  

        ### Ejemplos:  
        #### Ejemplo 1:
        Texto: "El paciente Ignacio Rico Pedroza fue atendido por el Dr. Ignacio Rubio Tortosa del Servicio de Urología, NºCol: 46 28 52938."
        Salida esperada:  
        NOMBRE_PACIENTE: Ignacio  
        APELLIDOS_PACIENTE: Rico Pedroza  
        MEDICO: Ignacio Rubio Tortosa  

        #### Ejemplo 2:
        Texto: "Francisco Javier Serra Ortega acudió a consulta con el doctor José Antonio Cánovas Ivorra."
        Salida esperada:  
        NOMBRE_PACIENTE: Francisco Javier  
        APELLIDOS_PACIENTE: Serra Ortega  
        MEDICO: José Antonio Cánovas Ivorra  

        Texto: {text}
        """,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}

    # Hacer la solicitud POST al servidor local
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        # Parsear la respuesta del modelo
        return parse_llm_response(response.json()['response'])
    else:
        raise Exception(f"Error en la petición al servidor: {response.status_code}")


def parse_llm_response(response):
    """
    Parsea la respuesta del modelo LLM para extraer las entidades nombradas.
    """
    print(response)
    entities = {
        'NOMBRE_PACIENTE': None,
        'APELLIDOS_PACIENTE': None,
        'MEDICO': None
    }

    for line in response.split('\n'):
        if line.startswith('NOMBRE_PACIENTE:'):
            entities['NOMBRE_PACIENTE'] = line.replace('NOMBRE_PACIENTE:', '').strip()
        elif line.startswith('APELLIDOS_PACIENTE:'):
            entities['APELLIDOS_PACIENTE'] = line.replace('APELLIDOS_PACIENTE:', '').strip()
        elif line.startswith('MEDICO:'):
            entities['MEDICO'] = line.replace('MEDICO:', '').strip()

    return entities


def generate_xml_output(text, entities):
    """
    Genera el XML con las entidades nombradas y identificadores fijos.
    """
    # Crear el elemento raíz
    meddocan = ET.Element('MEDDOCAN')

    # Crear el elemento TEXT
    text_element = ET.SubElement(meddocan, 'TEXT')
    text_element.text = f"<![CDATA[{text}]]>"

    # Crear el elemento TAGS
    tags_element = ET.SubElement(meddocan, 'TAGS')

    # Añadir las entidades nombradas con identificadores fijos
    if entities['NOMBRE_PACIENTE']:
        # Nombre del paciente (T21)
        name_element = ET.SubElement(tags_element, 'NAME')
        name_element.set('id', 'T21')
        name_element.set('start', str(text.find(entities['NOMBRE_PACIENTE'])))
        name_element.set('end', str(text.find(entities['NOMBRE_PACIENTE']) + len(entities['NOMBRE_PACIENTE'])))
        name_element.set('text', entities['NOMBRE_PACIENTE'])
        name_element.set('TYPE', 'NOMBRE_SUJETO_ASISTENCIA')
        name_element.set('comment', '')

    if entities['APELLIDOS_PACIENTE']:
        # Apellidos del paciente (T20)
        name_element = ET.SubElement(tags_element, 'NAME')
        name_element.set('id', 'T20')
        name_element.set('start', str(text.find(entities['APELLIDOS_PACIENTE'])))
        name_element.set('end', str(text.find(entities['APELLIDOS_PACIENTE']) + len(entities['APELLIDOS_PACIENTE'])))
        name_element.set('text', entities['APELLIDOS_PACIENTE'])
        name_element.set('TYPE', 'NOMBRE_SUJETO_ASISTENCIA')
        name_element.set('comment', '')

    if entities['MEDICO']:
        # Nombre y apellidos del médico (T9)
        name_element = ET.SubElement(tags_element, 'NAME')
        name_element.set('id', 'T9')
        name_element.set('start', str(text.find(entities['MEDICO'])))
        name_element.set('end', str(text.find(entities['MEDICO']) + len(entities['MEDICO'])))
        name_element.set('text', entities['MEDICO'])
        name_element.set('TYPE', 'NOMBRE_PERSONAL_SANITARIO')
        name_element.set('comment', '')

    # Convertir el árbol XML a una cadena
    xml_str = ET.tostring(meddocan, encoding='utf-8').decode('utf-8')

    return xml_str


def process_xml_files(input_dir, output_dir):
    """
    Procesa los archivos XML usando el modelo LLM.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.xml'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
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

                # Generar nuevo XML
                output_xml = generate_xml_output(text, entities)

                # Guardar el resultado
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                    f.write(output_xml)

                print(f"Procesado y guardado: {output_path}")

            except ET.ParseError as e:
                print(f"Error de parseo en {filename}: {e}")
            except ValueError as e:
                print(f"Error en {filename}: {e}")
            except Exception as e:
                print(f"Error inesperado en {filename}: {e}")


if __name__ == "__main__":
    input_directory = 'test/xml'
    output_directory = 'output/xml'
    process_xml_files(input_directory, output_directory)