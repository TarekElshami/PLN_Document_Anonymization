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
Dado un texto clínico, identifica y anota **todas las entidades de información sensible** (ISP) siguiendo las directrices **estrictas** de la guía oficial de anotación del Plan de Impulso de las Tecnologías del Lenguaje para información de salud protegida.

🎯 Objetivos:
1. Detectar TODAS las menciones de información sensible en el texto.
2. Etiquetarlas usando esta sintaxis: <<<ETIQUETA>>>texto<<</ETIQUETA>>>
3. Extraer dichas entidades agrupadas por tipo en un JSON válido.

🔖 CATEGORÍAS DE ETIQUETAS (USA SOLO ESTAS, NO INVENTES NI OMITAS)
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

⚠️ ACLARACIONES CRÍTICAS (Errores comunes evitables)
- **Nombres de médicos, enfermeros o personal clínico** deben etiquetarse como `NOMBRE_PERSONAL_SANITARIO`, **nunca** como `NOMBRE_SUJETO_ASISTENCIA`. No incluyas "Dr.", "Dra.", etc., dentro de la etiqueta.
- Cuando haya **varios formatos de una misma entidad** (ej. "3 años" y "tres años"), **anótalos todos** por separado, no ignores duplicados semánticos.
- La **edad o datos de familiares** (ej. "el hermano tiene seis años") deben anotarse como `FAMILIARES_SUJETO_ASISTENCIA`, **no** como edad del paciente.
- Reconoce **todas las formas de expresar el sexo** del paciente: M, F, varón, mujer, niño, niña, masculino, femenino…
- **Nunca etiquetes el nombre de un profesional clínico como paciente**.

⚠️ ERRORES CRÍTICOS A EVITAR:
1. NUNCA etiquetes "Dr." o "Dra." como parte del nombre del personal sanitario
2. NUNCA etiquetes un nombre de médico como NOMBRE_SUJETO_ASISTENCIA
3. NUNCA confundas los datos de un familiar con datos del paciente
4. NUNCA crees categorías que no estén en la lista como EDAD_FAMILIAR
5. NUNCA devuelvas "NINGUNA" como etiqueta si no hay entidades
6. SIEMPRE anota TODAS las menciones de la misma entidad aunque aparezcan varias veces

IMPORTANTE: Las etiquetas SIEMPRE deben seguir EXACTAMENTE este formato, sin excepciones:
<<<ETIQUETA>>>valor<<</ETIQUETA>>>
- Usa exactamente 3 símbolos < para abrir (<<<)
- Usa exactamente 3 símbolos > para cerrar (>>>)
- El cierre SIEMPRE es <<</ETIQUETA>>> con exactamente 3 símbolos < y 3 símbolos >
- NO uses otras variantes como <<<<, <<</, <<<[/, etc.

🧾 Reglas de anotación (estrictas y actualizadas)
1. No etiquetes campos de formulario (“Nombre:”, “Edad:”, etc.).
2. No incluyas signos de puntuación, tildes ni espacios dentro de las etiquetas.
3. No repitas una entidad si ya está anotada, **salvo si aparece en distintas formas o contextos**.
4. Múltiples palabras que constituyen una sola entidad deben etiquetarse juntas.
5. Elimina títulos o prefijos ("Dr.", "Dña.") **fuera** de la etiqueta.
6. Toda información sensible debe ser etiquetada si aparece, sin omitir ninguna.
7. **Si no hay entidades sensibles**, devuelve el JSON con `"entidades": {{}}`.
8. **No generes etiquetas nuevas ni marques datos que no figuren en la lista.**
9. Sigue las reglas multipalabra, alias, abreviaciones y convenciones de las guías oficiales.
10. Usa el sentido del contexto para distinguir entidades del paciente vs. familiares.

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

# Diccionario de límites de contexto por modelo
MODEL_CONTEXT_LIMITS = {
    "llama3.2:1B": 2048,
    "llama3.3": 2048
}

SAFETY_MARGIN = 0.1  # 10% de margen de seguridad


def build_meddocan_xml(original_text, tagged_text):
    """
    Construye un XML estilo MEDDOCAN capturando TODAS las entidades, incluso duplicadas.
    """
    root = ET.Element("MEDDOCAN")
    text_elem = ET.SubElement(root, "TEXT")
    text_elem.text = original_text
    tags_elem = ET.SubElement(root, "TAGS")

    pattern = r"<<<(.*?)>>>(.*?)<<</\1>>>"
    last_end = 0  # Track para búsquedas incrementales

    for match in re.finditer(pattern, tagged_text):
        entity_type = match.group(1)
        entity_text = match.group(2).strip()  # Limpiar espacios
        xml_tag = TAG_CATEGORIES.get(entity_type, "WARNING")

        # Buscar la entidad DESPUÉS de la última posición registrada
        try:
            start = original_text.index(entity_text, last_end)
            end = start + len(entity_text)
            last_end = end
        except ValueError:
            print(f"⚠️ Entidad no encontrada: '{entity_text}' (posible error de alineación)")
            continue

        # Verificar que el texto coincida EXACTAMENTE
        if original_text[start:end] != entity_text:
            print(f"⚠️ Discrepancia en: '{entity_text}' vs '{original_text[start:end]}'")
            continue

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

    print("\n" + "=" * 50)
    print("Iniciando división del texto en fragmentos...")
    print(f"Líneas totales: {len(lines)}")
    print(f"Límite seguro de tokens por fragmento: {max_safe_tokens}")
    print("=" * 50 + "\n")

    for i, line in enumerate(lines):
        if line.strip() == "":
            continue  # Ignorar líneas vacías

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
                print(f"\nFragmento completado ({len(chunks)} fragmentos hasta ahora)")
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