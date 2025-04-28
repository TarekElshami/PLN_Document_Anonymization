from xml.dom import minidom
import requests
import xml.etree.ElementTree as ET
import os
import tiktoken
import ast
import re

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
    'CENTRO_SALUD': 'LOCATION',

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
Dado un texto clinico, anota todas las entidades sensibles de acuerdo con las categorías definidas en la guía oficial de anotación de información de salud protegida.

Objetivos
1. Identificar menciones explícitas de información sensible (ISP) contenidas en un texto médico.
2. Etiquetar dichas menciones utilizando tags personalizados con la forma:
<<<ETIQUETA>>>texto<<</ETIQUETA>>>
3. Extraer las entidades anotadas junto con su categoría.

Etiquetas a utilizar
Estas etiquetas corresponden a las definiciones oficiales del plan de anotación de información de salud protegida y cubren todas las categorías relevantes:

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
CALLE: Dirección postal completa, incluyendo tipo de vía, nombre, número, piso, etc.
TERRITORIO: Ciudad, provincia, código postal, barrio, comarca, o cualquier otra división geográfica.
PAIS: País mencionado en el texto.
CORREO_ELECTRONICO: Cualquier dirección de correo electrónico.
NUMERO_TELEFONO: Números de teléfono personales o profesionales.
NUMERO_FAX: Números de fax asociados a la atención o el paciente.
DIREC_PROT_INTERNET: Direcciones de protocolo de Internet (IP, TCP, SMTP, etc.).
URL_WEB: Cualquier dirección web o enlace.
PROFESION: Profesión del paciente o familiares.
HOSPITAL: Nombres de hospitales o centros sanitarios.
ID_CENTRO DE SALUD: Nombres de centros de salud o unidades clínicas.
INSTITUCION: Cualquier otra institución no médica identificable.
NUMERO_IDENTIF: Otros números de identificación no clasificados.
IDENTIF_VEHICULOS_NRSERIE_PLACAS: Matrículas o números de bastidor de vehículos.
IDENTIF_DISPOSITIVOS_NRSERIE: Identificadores de dispositivos médicos (serie, chip, etc.).
IDENTIF_BIOMETRICOS: Huellas, escaneos o cualquier identificador biométrico.
OTROS_SUJETO_ASISTENCIA: Cualquier información adicional que pueda permitir la identificación del paciente y no esté incluida en las categorías anteriores.

🧾 Reglas de anotación estrictas
1. No anotar etiquetas o claves del formulario (como "Nombre:", "Edad:", etc.).
2. No incluir espacios ni signos de puntuación ni tildes dentro de las etiquetas.
3. Una etiqueta por entidad, aunque se repita en el texto.
4. Etiquetar múltiples palabras como una sola mención si pertenecen a la misma categoría y están juntas.
5. Excluir títulos o prefijos como "Dr.", "Dña." de las etiquetas de nombres.
6. Anotar todas las fechas, edades, lugares y contactos que puedan identificar al paciente o profesional.
7. Si una etiqueta no tiene ninguna entidad entonces no debes mencionarla
8. No debes inventarte una etiqueta que no esté en esa lista
9. Si ves información sensible que no esté en la lista de etiquetas no debes mencionarla
10. Si no encuentras ninguna información sensible simplemente devuelve unas entidades vacias y ya
11. Nunca etiquetar el nombre de médicos o personal sanitario como NOMBRE_SUJETO_ASISTENCIA. Médicos/enfermeros/técnicos deben ir en NOMBRE_PERSONAL_SANITARIO.

🧪 Entrada esperada
Cualquier texto clínico en formato libre.

✅ Salida esperada
Devuélveme ÚNICAMENTE un JSON válido. Sin explicaciones, sin introducción, sin comentarios y con la siguiente estructura:
{{
  "texto_anotado": "Texto clínico con etiquetas <<<ETIQUETA>>>...<<</ETIQUETA>>> ya insertadas",
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

# Diccionario de límites de contexto por modelo
MODEL_CONTEXT_LIMITS = {
    "llama3.2:1B": 2048,
    "llama3.3": 2048
}

SAFETY_MARGIN = 0.1  # 10% de margen de seguridad


def build_meddocan_xml(original_text, tagged_text):
    """
    Construye un XML estilo MEDDOCAN a partir del texto original, el texto anotado y las entidades.
    """

    # Inicializar XML
    root = ET.Element("MEDDOCAN")
    text_elem = ET.SubElement(root, "TEXT")
    text_elem.text = original_text

    tags_elem = ET.SubElement(root, "TAGS")

    # Buscar entidades en el texto anotado para ubicar sus posiciones reales en el texto original
    pattern = r"<<<(.*?)>>>(.*?)<<</\1>>>"


    for match in re.finditer(pattern, tagged_text):
        entity_type = match.group(1)
        entity_text = match.group(2)

        # Mapea a la etiqueta general (e.g., NAME, AGE...)
        xml_tag = TAG_CATEGORIES.get(entity_type, "WARNING")

        # Encontrar en el texto original (segura para duplicados)
        try:
            start = original_text.index(entity_text)
        except ValueError:
            print(f"⚠️ No se pudo encontrar '{entity_text}' en el texto original. Saltando esta entidad.")
            continue

        end = start + len(entity_text)

        ET.SubElement(tags_elem, xml_tag, {
            "start": str(start),
            "end": str(end),
            "text": entity_text,
            "TYPE": entity_type
        })

    return ET.tostring(root, encoding="unicode", method="xml")


def get_gbnf_grammar():
    etiquetas = list(TAG_CATEGORIES.keys())
    etiquetas_union = " | ".join(f'"{tag}"' for tag in etiquetas)

    return f"""
root ::= '{{' '"texto_anotado":' string ',' '"entidades":' '{{' entidad (',' entidad)* '}}' '}}'

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

    print("\n" + "=" * 50)
    print("Iniciando división del texto en fragmentos...")
    print(f"Líneas totales: {len(lines)}")
    print(f"Límite seguro de tokens por fragmento: {max_safe_tokens}")
    print("=" * 50 + "\n")

    for i, line in enumerate(lines):
        temp_chunk = '\n'.join(current_chunk + [line])
        prompt_with_chunk = PROMPT_BASE.format(texto_clinico=temp_chunk)
        tokens = encoding.encode(prompt_with_chunk)

        print(f"Línea {i + 1}: Tokens totales del actual fragmento= {len(tokens)}", end=" | ")

        if len(tokens) < max_safe_tokens:
            current_chunk.append(line)
            print("Añadida al fragmento actual")
        else:
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                print(f"\n⚠️ Fragmento completado ({len(chunks)} fragmentos hasta ahora)")
                print(f"Contenido del fragmento:\n{'-' * 30}\n{chunks[-1]}\n{'-' * 30}\n")
                current_chunk = [line]
            else:
                # Línea demasiado larga para el prompt incluso sola
                chunks.append(line)
                print(
                    f"\n⚠️ Línea individual demasiado larga, creando fragmento especial ({len(chunks)} fragmentos hasta ahora)")
                print(f"Contenido del fragmento:\n{'-' * 30}\n{chunks[-1]}\n{'-' * 30}\n")
                current_chunk = []

    if current_chunk:
        chunks.append('\n'.join(current_chunk))
        print(f"\nÚltimo fragmento completado ({len(chunks)} fragmentos en total)")
        print(f"Contenido del fragmento:\n{'-' * 30}\n{chunks[-1]}\n{'-' * 30}\n")

    print("\n" + "=" * 50)
    print(f"División completada. Total de fragmentos creados: {len(chunks)}")
    print("=" * 50 + "\n")

    return chunks


def extract_entities(text):
    """
    Procesa el texto con el modelo, dividiéndolo si excede el límite de tokens.
    Devuelve el texto anotado y todas las entidades combinadas.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    max_tokens = MODEL_CONTEXT_LIMITS.get(MODEL_NAME, 0)
    safe_max_tokens = int(max_tokens * (1 - SAFETY_MARGIN))

    prompt = PROMPT_BASE.format(texto_clinico=text)
    total_tokens = len(encoding.encode(prompt))

    print("\n" + "=" * 50)
    print("Iniciando extracción de entidades...")
    print(f"Tokens totales del texto completo: {total_tokens}")
    print(f"Límite seguro de tokens: {safe_max_tokens}")
    print("=" * 50 + "\n")

    # Determinar si hay que dividir
    if total_tokens > safe_max_tokens:
        print("⚠️ Dividiendo el texto por líneas debido al límite de tokens...")
        text_parts = split_text_by_newline(text, safe_max_tokens)
    else:
        text_parts = [text]
        print("✅ El texto cabe en un solo fragmento, no es necesario dividir")

    all_tagged = []
    all_entities = {}

    for idx, part in enumerate(text_parts):
        print("\n" + "=" * 50)
        print(f"Procesando fragmento {idx + 1}/{len(text_parts)}")
        print(f"Contenido del fragmento:\n{'-' * 30}\n{part}\n{'-' * 30}")
        print("=" * 50 + "\n")

        payload = {
            "model": MODEL_NAME,
            "prompt": PROMPT_BASE.format(texto_clinico=part),
            "stream": False,
            "grammar": get_gbnf_grammar()
        }

        try:
            print("Enviando solicitud al modelo LLM...")
            response = requests.post("http://localhost:20201/api/generate", json=payload,
                                     headers={"Content-Type": "application/json"}, timeout=1000)
            response.raise_for_status()
            result = response.json()['response']

            print("\n" + "=" * 50)
            print(f"Respuesta del LLM para fragmento {idx + 1}:")
            print(f"{'-' * 30}\n{result}\n{'-' * 30}")
            print("=" * 50 + "\n")

            # Intentamos parsear el JSON del modelo y acumular entidades
            try:
                parsed = ast.literal_eval(result)
                fragment_entities = parsed.get("entidades", {})

                print(f"Entidades encontradas en fragmento {idx + 1}:")
                for tag, values in fragment_entities.items():
                    print(f"- {tag}: {values}")
                    if tag not in all_entities:
                        all_entities[tag] = set()
                    all_entities[tag].update(values)

                all_tagged.append(parsed.get("texto_anotado", part))
            except Exception as parse_error:
                print(f"⚠️ Error parseando JSON en fragmento {idx + 1}: {parse_error}")
                print(f"Respuesta cruda del modelo:\n{result}")
                all_tagged.append(part)

        except requests.exceptions.RequestException as e:
            print(f"❌ Error al procesar fragmento {idx + 1}: {e}")
            all_tagged.append(part)

    # Convertimos sets a listas para salida limpia
    all_entities = {k: list(v) for k, v in all_entities.items()}

    print("\n" + "=" * 50)
    print("Resumen final de entidades encontradas:")
    for tag, values in all_entities.items():
        print(f"- {tag}: {values}")
    print("=" * 50 + "\n")

    return {
        'tagged_text': "\n\n".join(all_tagged),
        'entities': all_entities
    }


def process_xml_files(input_dir, output_dir):
    """Procesa archivos XML manteniendo ambas salidas."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in [f for f in os.listdir(input_dir) if f.endswith('.xml')]:
        try:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print("\n" + "=" * 80)
            print(f"Procesando archivo: {filename}")
            print("=" * 80 + "\n")

            # Leer XML original
            tree = ET.parse(input_path)
            original_text = tree.find('.//TEXT').text or ""

            print("Texto original extraído del XML:")
            print(f"{'-' * 30}\n{original_text}\n{'-' * 30}\n")

            # Extraer datos (texto marcado + entidades)
            result = extract_entities(original_text)

            # Mostrar resultados en consola
            print("\nTexto marcado final:")
            print(f"{'-' * 30}\n{result['tagged_text']}\n{'-' * 30}\n")

            # Crear XML estilo MEDDOCAN
            meddocan_xml = build_meddocan_xml(original_text, result['tagged_text'])

            # Embellecer XML si lo deseas
            pretty_xml = minidom.parseString(meddocan_xml).toprettyxml(indent="  ")

            # Guardar XML en salida
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)

            print(f"\n✅ Resultado guardado en: {output_path}")
            print("*" * 80)

        except Exception as e:
            print(f"❌ Error procesando {filename}: {str(e)}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("INICIANDO PROCESAMIENTO DE ARCHIVOS XML")
    print("=" * 80 + "\n")

    process_xml_files('test/xml', 'output/xml/LLaMA_model3.3')

    print("\n" + "=" * 80)
    print("PROCESAMIENTO COMPLETADO")
    print("=" * 80 + "\n")