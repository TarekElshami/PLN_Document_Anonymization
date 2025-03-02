import requests
import xml.etree.ElementTree as ET
import os
import re

def extract_entities(text):
    """
    Usa el modelo LLM en el servidor para identificar entidades nombrables en el texto.
    """
    url = "http://localhost:20201/api/generate"
    payload = {
        "model": "llama3.2:1b",
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
    # Asegurar que solo queden letras, espacios entre palabras, y puntos de abreviaturas
    cleaned_output = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s.-]', '', cleaned_output)
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
    output_directory = 'output/xml/human_crafted'
    process_xml_files(input_directory, output_directory)