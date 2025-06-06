import os
import xml.etree.ElementTree as ET
import tiktoken
import requests
import logging
import json
import ast
import time

# Configuración de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logAnonymizer.log', mode='w')
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

# Prompt por defecto para la detección de entidades
DEFAULT_PROMPT = """
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
7. **Si no hay entidades sensibles**, devuelve el JSON con "entidades": {{}}.
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
Devuélveme solo el JSON como texto plano, sin comillas invertidas, sin markdown ni delimitadores de código. No uses etiquetas tipo json ni ningún bloque de código.
El texto que debes analizar es este:
 "{texto_clinico}"
"""

# Configuración por defecto
DEFAULT_MODEL_NAME = "llama3.3"
DEFAULT_OLLAMA_PORT = "20201"
DEFAULT_SAFETY_MARGIN = 0.1  # 10% de margen de seguridad

# Diccionario de límites de contexto por modelo
MODEL_CONTEXT_LIMITS = {
    "llama3.2:1B": 2048,
    "llama3.3": 2048,
    "phi4": 2048,
    "qwen3:1.7b": 2048,
    "qwen3:235b": 2048,
    "deepseek-r1:70b": 2048,
    "deepseek-r1:1.5b": 2048,
}


class LLMClient:
    """Cliente para interactuar con un LLM y procesar textos clínicos."""

    def __init__(self, model_name=DEFAULT_MODEL_NAME, ollama_port=DEFAULT_OLLAMA_PORT,
                 prompt_base=DEFAULT_PROMPT, safety_margin=DEFAULT_SAFETY_MARGIN,
                 use_grammar=True):
        self.model_name = model_name
        self.ollama_port = ollama_port
        self.prompt_base = prompt_base
        self.safety_margin = safety_margin
        self.use_grammar = use_grammar
        self.max_retries = 3

        if model_name not in MODEL_CONTEXT_LIMITS:
            raise ValueError(f"Modelo '{model_name}' no encontrado en MODEL_CONTEXT_LIMITS. "
                             f"Modelos disponibles: {list(MODEL_CONTEXT_LIMITS.keys())}")

        self.max_tokens = MODEL_CONTEXT_LIMITS[model_name]
        self.safe_max_tokens = int(self.max_tokens * (1 - safety_margin))
        self.encoding = tiktoken.get_encoding("cl100k_base")

        logger.info(f"Cliente LLM inicializado con modelo: {model_name}")
        logger.info(f"Puerto Ollama: {ollama_port}")
        logger.info(f"Límite de contexto: {self.max_tokens} tokens")
        logger.info(f"Límite seguro (con margen del {safety_margin * 100}%): {self.safe_max_tokens} tokens")

    def get_gbnf_grammar(self):
        """Genera la gramática GBNF para restringir la salida del modelo."""
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

    def split_text_by_newline(self, text):
        """
        Divide el texto por saltos de línea para mantener cada fragmento bajo el límite de tokens.

        Args:
            text (str): Texto completo a dividir.

        Returns:
            list: Lista de fragmentos de texto.
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []

        logger.info("Iniciando división del texto en fragmentos...")
        logger.info(f"Líneas totales: {len(lines)}")
        logger.info(f"Límite seguro de tokens por fragmento: {self.safe_max_tokens}")

        for i, line in enumerate(lines):
            if line.strip() == "":
                continue  # Ignorar líneas vacías

            temp_chunk = '\n'.join(current_chunk + [line])
            prompt_with_chunk = self.prompt_base.format(texto_clinico=temp_chunk)
            tokens = self.encoding.encode(prompt_with_chunk)

            logger.debug(f"Línea {i + 1}: Tokens totales del actual fragmento= {len(tokens)}")

            if len(tokens) < self.safe_max_tokens:
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

    def query_llm(self, text_fragment):
        """
        Envía una consulta al LLM para procesar un fragmento de texto.

        Args:
            text_fragment (str): Fragmento de texto a procesar.

        Returns:
            str: Respuesta cruda del LLM en formato JSON.
        """
        prompt = self.prompt_base.format(texto_clinico=text_fragment)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": self.max_tokens
            }
        }

        if self.use_grammar:
            payload["grammar"] = self.get_gbnf_grammar()

        try:
            logger.debug("Enviando solicitud al modelo LLM...")
            response = requests.post(
                f"http://localhost:{self.ollama_port}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=1000
            )
            response.raise_for_status()
            result = response.json()['response']

            logger.debug(f"Respuesta cruda del LLM:\n{result}")
            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Error en la solicitud al modelo: {e}")
            return ""

    def extract_entities(self, text):
        """
        Procesa el texto con el modelo, dividiéndolo si excede el límite de tokens.
        Devuelve el texto anotado y todas las entidades combinadas.
        """
        total_tokens = len(self.encoding.encode(self.prompt_base.format(texto_clinico=text)))

        logger.info("Iniciando extracción de entidades...")
        logger.info(f"Tokens totales del texto completo: {total_tokens}")

        # Determinar si hay que dividir
        if total_tokens > self.safe_max_tokens:
            logger.warning("Dividiendo el texto por líneas debido al límite de tokens...")
            text_parts = self.split_text_by_newline(text)
        else:
            text_parts = [text]
            logger.info("El texto cabe en un solo fragmento, no es necesario dividir")

        all_tagged = []
        all_entities = {}

        for idx, part in enumerate(text_parts):
            logger.info(f"Procesando fragmento {idx + 1}/{len(text_parts)}")
            logger.debug(f"Contenido del fragmento:\n{'-' * 30}\n{part}\n{'-' * 30}")

            parsed = None
            retries = 0
            while retries < self.max_retries:
                llm_output = self.query_llm(part)

                try:
                    # Use ast.literal_eval for safer parsing of string representations of Python literals
                    parsed = ast.literal_eval(llm_output)
                    logger.info(f"JSON parseado exitosamente en el intento {retries + 1} para fragmento {idx + 1}.")
                    break  # Exit retry loop if successful
                except Exception as parse_error:
                    logger.warning(
                        f"Error parseando JSON en fragmento {idx + 1} (intento {retries + 1}/{self.max_retries}): {parse_error}")
                    logger.warning(f"Respuesta cruda del modelo:\n{llm_output}")
                    retries += 1
                    if retries < self.max_retries:
                        logger.info(f"Reintentando procesar fragmento {idx + 1} en {2 ** retries} segundos...")
                        time.sleep(2 ** retries)  # Exponential back-off for retries
                    else:
                        logger.error(
                            f"Fallo al parsear JSON después de {self.max_retries} reintentos para fragmento {idx + 1}. Procesando sin entidades para este fragmento.")
                        parsed = {"texto_anotado": part,
                                  "entidades": {}}  # Fallback to original part and empty entities

            fragment_entities = parsed.get("entidades", {})

            logger.info(f"Entidades encontradas en fragmento {idx + 1}:")
            for tag, values in fragment_entities.items():
                logger.info(f"- {tag}: {values}")
                if tag not in all_entities:
                    all_entities[tag] = set()
                all_entities[tag].update(values)

            all_tagged.append(parsed.get("texto_anotado", part))

        # Convertimos sets a listas para salida limpia
        all_entities = {k: list(v) for k, v in all_entities.items()}

        logger.info("Resumen final de entidades encontradas:")
        for tag, values in all_entities.items():
            logger.info(f"- {tag}: {values}")

        return {
            'tagged_text': "\n\n".join(all_tagged),
            'entities': all_entities
        }

    def process_text_file(self, input_text, output_path=None):
        """
        Procesa un texto plano y guarda la salida procesada (JSON) en un archivo.
        """
        result = self.extract_entities(input_text)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Resultados procesados guardados en: {output_path}")

        return result


# Función de utilidad para uso independiente
def process_llm_request(input_file_path, output_dir, ollama_port, model_name, prompt_text, use_grammar, context_size):
    try:
        # Leer el archivo de entrada
        tree = ET.parse(input_file_path)
        original_text = tree.find('.//TEXT').text or ""

        # Update MODEL_CONTEXT_LIMITS for the current run
        MODEL_CONTEXT_LIMITS[model_name] = context_size

        # Configurar el cliente LLM
        client = LLMClient(
            model_name=model_name,
            ollama_port=ollama_port,
            prompt_base=prompt_text,
            use_grammar=use_grammar
        )

        # Procesar el texto
        result = client.extract_entities(original_text)

        # Crear el nombre del archivo de salida
        input_basename = os.path.basename(input_file_path)
        output_filename = os.path.splitext(input_basename)[0]
        output_path = os.path.join(output_dir, f"{output_filename}_llm_response.txt")

        # Guardar los resultados
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logging.info(f"Procesamiento completado exitosamente. Resultados guardados en: {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error durante el procesamiento: {str(e)}")
        return False