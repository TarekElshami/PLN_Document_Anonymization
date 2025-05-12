from xml.dom import minidom
import requests
import xml.etree.ElementTree as ET
import os
import tiktoken
import ast
import argparse
from collections import Counter
import logging
from tqdm import tqdm

# Configuración de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anonymization.log'),
        # logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Mapeo de etiquetas específicas a categorías generales
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

PROMPT_BASE = """
Tarea:
Dado un texto clínico, identifica y anota **todas las entidades de información sensible protegida** (ISP) siguiendo las directrices **estrictas** de la guía oficial de anotación del Plan de Impulso de las Tecnologías del Lenguaje para información de salud protegida.

🎯 Objetivos:
1. Detectar TODAS las menciones de información sensible en el texto.
2. Etiquetarlas usando esta sintaxis: <<<ETIQUETA>>>texto<<</ETIQUETA>>>
3. Extraer dichas entidades agrupadas por tipo en un JSON válido.

CATEGORÍAS DE ETIQUETAS (USA SOLO ESTAS, NO INVENTES NINGUNA)
NOMBRE_SUJETO_ASISTENCIA: Solo el nombre y apellidos del paciente. También iniciales, apodos o motes.
NOMBRE_PERSONAL_SANITARIO: Nombre y apellidos de médicos, enfermeros, técnicos u otro personal clínico.
FAMILIARES_SUJETO_ASISTENCIA: Nombres, apellidos o datos personales de familiares del paciente (edad, parentesco, número).
ID_SUJETO_ASISTENCIA: Códigos como NHC, CIPA, DNI, NIF, pasaporte u otros identificadores del paciente.
ID_TITULACION_PERSONAL_SANITARIO: Número de colegiado o licencia del profesional sanitario.
ID_CONTACTO_ASISTENCIAL: Identificador de episodios clínicos o procesos.
ID_ASEGURAMIENTO: Número de afiliación a la seguridad social (NASS).
EDAD_SUJETO_ASISTENCIA: Edad del paciente (incluyendo formas como "tres días", "6 años").
SEXO_SUJETO_ASISTENCIA: Sexo del paciente (incluyendo formas como "varón", "niña", "M", "H").
FECHAS: Cualquier fecha del calendario (de nacimiento, ingreso, evolución, etc.).
CALLE: Dirección postal completa, incluyendo tipo de vía, nombre, número, piso, etc..
TERRITORIO: Ciudad, provincia, código postal, barrio, comarca, o cualquier otra división geográfica.
PAIS: País mencionado en el texto.
CORREO_ELECTRONICO: Cualquier dirección de correo electrónico.
NUMERO_TELEFONO: Números de teléfono personales o profesionales.
NUMERO_FAX: Números de fax asociados a la atención o el paciente.
DIREC_PROT_INTERNET: Direcciones de protocolo de Internet (IP, TCP, SMTP, etc.).
URL_WEB: Cualquier dirección web o enlace.
PROFESION: Profesión del paciente o familiares.
HOSPITAL: Nombres de hospitales o centros sanitarios.
ID_CENTRO_DE_SALUD: Nombres de centros de salud o unidades clínicas.
INSTITUCION: Cualquier otra institución no médica identificable.
NUMERO_IDENTIF: Otros números de identificación no clasificados.
IDENTIF_VEHICULOS_NRSERIE_PLACAS: Matrículas o números de bastidor de vehículos.
IDENTIF_DISPOSITIVOS_NRSERIE: Identificadores de dispositivos médicos (serie, chip, etc.).
IDENTIF_BIOMETRICOS: Huellas, escaneos o cualquier identificador biométrico.
OTROS_SUJETO_ASISTENCIA: Cualquier información adicional que pueda permitir la identificación del paciente y no esté incluida en las categorías anteriores.

⚠️ ACLARACIONES
- Cuando haya **varios formatos de una misma entidad** (ej. "3 años" y "tres años"), **anótalos todos** por separado, no ignores duplicados semánticos.
- Reconoce **todas las formas de expresar el sexo** del paciente: M, F, varón, mujer, niño, niña, masculino, femenino…

🧾 Reglas de anotación (estrictas y actualizadas)
1. No incluir dentro de las etiquetas claves del formulario como "Nombre:", "Edad:", etc..
2. No incluir espacios ni signos de puntuación dentro de las etiquetas.
3. Debes incluir todas las apariciones de las entidades aunque se repitan
4. Múltiples palabras que constituyen una sola entidad deben etiquetarse juntas.
5. Dejar fuera de las etiquetas títulos o prefijos como "Dr.", "Dña.".
6. Toda información sensible debe ser etiquetada si aparece, sin omitir ninguna.
7. **Si no hay entidades sensibles**, devuelve el JSON con `"entidades": {{}}`.
8. **No generes etiquetas nuevas ni marques datos que no figuren en la lista.**
9. Usa el contexto del documento para distinguir entre entidades del paciente vs. familiares.

✅ Salida esperada (formato obligatorio)
ÚNICAMENTE un **JSON válido**, sin explicaciones, sin formato de bloque de código ni comentarios. Estructura esperada:
{{
  "texto_anotado": "Texto con etiquetas <<<ETIQUETA>>>...<<</ETIQUETA>>> ya insertadas",
  "entidades": {{
    "ETIQUETA1": ["valor1", "valor2", ...],
    "ETIQUETA2": ["valor1", ...]
  }}
}}
Devuélveme solo el JSON como texto plano, sin comillas invertidas, sin markdown ni delimitadores de código. No uses etiquetas tipo ```json ni ningún bloque de código.
El texto que debes analizar es este:
 "{texto_clinico}"
"""

MODEL_NAME = "llama3.3"
DEFAULT_OLLAMA_PORT = "20201"

# Diccionario de límites de contexto por modelo
MODEL_CONTEXT_LIMITS = {
    "llama3.2:1B": 2048,
    "llama3.3": 2048
}

SAFETY_MARGIN = 0.1  # 10% de margen de seguridad

USE_GRAMMAR = True

def quitar_tildes(texto):
    reemplazos = {
        'á': 'a',
        'é': 'e',
        'í': 'i',
        'ó': 'o',
        'ú': 'u',
    }
    return ''.join(reemplazos.get(c, c) for c in texto)


def build_meddocan_xml_from_entities(original_text, entidades_json):
    root = ET.Element("MEDDOCAN")
    text_elem = ET.SubElement(root, "TEXT")
    text_elem.text = original_text
    tags_elem = ET.SubElement(root, "TAGS")

    tag_id = 1

    for tag, values in entidades_json.items():
        xml_tag = TAG_CATEGORIES.get(tag)
        if xml_tag is None:
            tag_sin_tildes = quitar_tildes(tag)
            xml_tag = TAG_CATEGORIES.get(tag_sin_tildes, "WARNING")
        value_counts = Counter(values)

        for value, count in value_counts.items():
            start_idx = 0
            occurrences = 0

            while occurrences < count:
                start = original_text.find(value, start_idx)
                if start == -1:
                    break
                end = start + len(value)
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

    return ET.tostring(root, encoding="unicode", method="xml")


def get_gbnf_grammar():
    etiquetas = list(TAG_CATEGORIES.keys())
    etiquetas_union = " | ".join(f'"{tag}"' for tag in etiquetas)

    return f"""
root ::= '{{' '"texto_anotado":' string ',' '"entidades":' '{{' (entidad (',' entidad)*)? '}}' '}}'

entidad ::= etiqueta ':' '[' valores ']'
etiqueta ::= {etiquetas_union}
valores ::= valor (',' valor)*
valor ::= string
string ::= '"' chars '"'
chars ::= char*
char ::= [^"\\n]
""".strip()


def split_text_by_newline(text, max_safe_tokens):
    """
    Divide el texto por saltos de línea para mantener cada fragmento bajo el límite de tokens.
    Devuelve una lista de fragmentos seguros.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    lines = text.split('\n')
    chunks = []
    current_chunk = []

    logger.info("Iniciando división del texto en fragmentos...")
    logger.info(f"Líneas totales: {len(lines)}")
    logger.info(f"Límite seguro de tokens por fragmento: {max_safe_tokens}")

    for i, line in enumerate(lines):
        if line.strip() == "":
            continue  # Ignorar líneas vacías

        temp_chunk = '\n'.join(current_chunk + [line])
        prompt_with_chunk = PROMPT_BASE.format(texto_clinico=temp_chunk)
        tokens = encoding.encode(prompt_with_chunk)

        logger.debug(f"Línea {i + 1}: Tokens totales del actual fragmento= {len(tokens)}")

        if len(tokens) < max_safe_tokens:
            current_chunk.append(line)
            logger.debug("Añadida al fragmento actual")
        else:
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                logger.info(f"Fragmento completado ({len(chunks)} fragmentos hasta ahora)")
                logger.debug(f"Contenido del fragmento:\n{'-' * 30}\n{chunks[-1]}\n{'-' * 30}")
                current_chunk = [line]
            else:
                # Línea demasiado larga para el prompt incluso sola
                chunks.append(line)
                logger.warning(
                    f"Línea individual demasiado larga, creando fragmento especial ({len(chunks)} fragmentos hasta ahora)")
                logger.debug(f"Contenido del fragmento:\n{'-' * 30}\n{chunks[-1]}\n{'-' * 30}")
                current_chunk = []

    if current_chunk:
        chunks.append('\n'.join(current_chunk))
        logger.info(f"Último fragmento completado ({len(chunks)} fragmentos en total)")
        logger.debug(f"Contenido del fragmento:\n{'-' * 30}\n{chunks[-1]}\n{'-' * 30}")

    logger.info(f"División completada. Total de fragmentos creados: {len(chunks)}")
    return chunks


def extract_entities(text, ollama_port):
    """
    Procesa el texto con el modelo, dividiéndolo si excede el límite de tokens.
    Devuelve el texto anotado y todas las entidades combinadas.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    if MODEL_NAME not in MODEL_CONTEXT_LIMITS:
        raise ValueError(f"Modelo '{MODEL_NAME}' no encontrado en MODEL_CONTEXT_LIMITS. "
                         f"Modelos disponibles: {list(MODEL_CONTEXT_LIMITS.keys())}")

    max_tokens = MODEL_CONTEXT_LIMITS[MODEL_NAME]
    safe_max_tokens = int(max_tokens * (1 - SAFETY_MARGIN))

    prompt = PROMPT_BASE.format(texto_clinico=text)
    total_tokens = len(encoding.encode(prompt))

    logger.info("Iniciando extracción de entidades...")
    logger.info(f"Modelo seleccionado: {MODEL_NAME}")
    logger.info(f"Context limit definido: {max_tokens} tokens")
    logger.info(f"Tokens totales del texto completo: {total_tokens}")
    logger.info(f"Límite seguro de tokens (con margen del {SAFETY_MARGIN * 100}%): {safe_max_tokens}")
    logger.info(f"Puerto Ollama utilizado: {ollama_port}")

    # Determinar si hay que dividir
    if total_tokens > safe_max_tokens:
        logger.warning("Dividiendo el texto por líneas debido al límite de tokens...")
        text_parts = split_text_by_newline(text, safe_max_tokens)
    else:
        text_parts = [text]
        logger.info("El texto cabe en un solo fragmento, no es necesario dividir")

    all_tagged = []
    all_entities = {}

    for idx, part in enumerate(text_parts):
        logger.info(f"Procesando fragmento {idx + 1}/{len(text_parts)}")
        logger.debug(f"Contenido del fragmento:\n{'-' * 30}\n{part}\n{'-' * 30}")

        payload = {
            "model": MODEL_NAME,
            "prompt": PROMPT_BASE.format(texto_clinico=part),
            "stream": False,
            "options": {
                "num_ctx": max_tokens
            }
        }

        if USE_GRAMMAR:
            payload["grammar"] = get_gbnf_grammar()

        try:
            logger.debug("Enviando solicitud al modelo LLM...")
            response = requests.post(f"http://localhost:{ollama_port}/api/generate", json=payload,
                                     headers={"Content-Type": "application/json"}, timeout=1000)
            response.raise_for_status()
            result = response.json()['response']

            logger.debug(f"Respuesta del LLM para fragmento {idx + 1}:")
            logger.debug(f"{'-' * 30}\n{result}\n{'-' * 30}")

            # Intentamos parsear el JSON del modelo y acumular entidades
            try:
                parsed = ast.literal_eval(result)
                fragment_entities = parsed.get("entidades", {})

                logger.info(f"Entidades encontradas en fragmento {idx + 1}:")
                for tag, values in fragment_entities.items():
                    logger.info(f"- {tag}: {values}")
                    if tag not in all_entities:
                        all_entities[tag] = set()
                    all_entities[tag].update(values)

                all_tagged.append(parsed.get("texto_anotado", part))
            except Exception as parse_error:
                logger.error(f"Error parseando JSON en fragmento {idx + 1}: {parse_error}")
                logger.error(f"Respuesta cruda del modelo:\n{result}")
                all_tagged.append(part)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error al procesar fragmento {idx + 1}: {e}")
            all_tagged.append(part)

    # Convertimos sets a listas para salida limpia
    all_entities = {k: list(v) for k, v in all_entities.items()}

    logger.info("Resumen final de entidades encontradas:")
    for tag, values in all_entities.items():
        logger.info(f"- {tag}: {values}")

    return {
        'tagged_text': "\n\n".join(all_tagged),
        'entities': all_entities
    }


def process_xml_files(input_dir, output_dir, ollama_port):
    """Procesa archivos XML manteniendo ambas salidas."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Obtener lista de archivos XML
    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]

    if not xml_files:
        logger.warning(f"No se encontraron archivos XML en el directorio de entrada: {input_dir}")
        return

    logger.info(f"Iniciando procesamiento de {len(xml_files)} archivos XML...")

    # Barra de progreso con tqdm
    for filename in tqdm(xml_files, desc="Procesando archivos", unit="archivo"):
        try:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            logger.info(f"Procesando archivo: {filename}")

            # Leer XML original
            tree = ET.parse(input_path)
            original_text = tree.find('.//TEXT').text or ""

            logger.debug("Texto original extraído del XML:")
            logger.debug(f"{'-' * 30}\n{original_text}\n{'-' * 30}")

            # Extraer datos (texto marcado + entidades)
            result = extract_entities(original_text, ollama_port)

            logger.debug("\nTexto marcado final:")
            logger.debug(f"{'-' * 30}\n{result['tagged_text']}\n{'-' * 30}")

            # Crear XML estilo MEDDOCAN
            meddocan_xml = build_meddocan_xml_from_entities(original_text, result['entities'])

            # Embellecer XML
            pretty_xml = minidom.parseString(meddocan_xml).toprettyxml(indent="  ")

            # Guardar XML en salida
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)

            logger.info(f"Resultado guardado en: {output_path}")

        except Exception as e:
            logger.error(f"Error procesando {filename}: {str(e)}")


def process_single_xml_file(input_file_path, output_dir, ollama_port, model_name, prompt_text, usegrammar):
    """Procesa un único archivo XML y devuelve True si se guardó correctamente, False en caso contrario.

    Args:
        input_file_path (str): Ruta completa al archivo XML de entrada
        output_dir (str): Directorio de salida donde guardar el resultado
        ollama_port (str): Puerto de Ollama a utilizar
        model_name (str): Nombre del modelo a utilizar (debe estar en MODEL_CONTEXT_LIMITS)
        prompt_text (str): Texto del prompt a utilizar para el procesamiento

    Returns:
        bool: True si el archivo se procesó y guardó correctamente, False si hubo algún error
    """
    try:
        # Validar que el modelo existe
        if model_name not in MODEL_CONTEXT_LIMITS:
            logger.error(f"Modelo '{model_name}' no encontrado en MODEL_CONTEXT_LIMITS")
            return False

        # Configurar el modelo global para esta ejecución
        global MODEL_NAME
        MODEL_NAME = model_name

        # Configurar el prompt base
        global PROMPT_BASE
        PROMPT_BASE = prompt_text

        global USE_GRAMMAR
        USE_GRAMMAR = usegrammar

        # Crear directorio de salida si no existe
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info(f"Procesando archivo: {input_file_path}")
        logger.info(f"Modelo: {model_name}")
        logger.info(f"Directorio de salida: {output_dir}")

        # Leer XML original
        tree = ET.parse(input_file_path)
        original_text = tree.find('.//TEXT').text or ""

        logger.debug("Texto original extraído del XML:")
        logger.debug(f"{'-' * 30}\n{original_text}\n{'-' * 30}")

        # Extraer datos (texto marcado + entidades)
        result = extract_entities(original_text, ollama_port)

        logger.debug("\nTexto marcado final:")
        logger.debug(f"{'-' * 30}\n{result['tagged_text']}\n{'-' * 30}")

        # Crear XML estilo MEDDOCAN
        meddocan_xml = build_meddocan_xml_from_entities(original_text, result['entities'])

        # Embellecer XML
        pretty_xml = minidom.parseString(meddocan_xml).toprettyxml(indent="  ")

        # Guardar XML en salida
        output_filename = os.path.basename(input_file_path)
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

        logger.info(f"Resultado guardado correctamente en: {output_path}")
        return True

    except ET.ParseError as e:
        logger.error(f"Error al parsear XML {input_file_path}: {str(e)}")
        return False
    except IOError as e:
        logger.error(f"Error de E/S al procesar {input_file_path}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error inesperado procesando {input_file_path}: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Procesador de archivos XML clínicos con Ollama')
    parser.add_argument('--port', type=str, default=DEFAULT_OLLAMA_PORT,
                        help=f'Puerto de Ollama (por defecto: {DEFAULT_OLLAMA_PORT})')

    args = parser.parse_args()

    logger.info("INICIANDO PROCESAMIENTO DE ARCHIVOS XML")
    logger.info(f"Usando puerto Ollama: {args.port}")

    process_xml_files('test/xml', 'output/xml/LLaMA_model3.3', args.port)

    logger.info("PROCESAMIENTO COMPLETADO")