import requests
import xml.etree.ElementTree as ET
import os
import tiktoken

# Mapeo de etiquetas específicas a categorías generales
TAG_CATEGORIES = {
    # NAME (Nombre)
    'NOMBRE_SUJETO_ASISTENCIA': 'NAME',
    'NOMBRE_PERSONAL_SANITARIO': 'NAME',

    # PROFESSION (Profesión)
    'PROFESION': 'PROFESSION',
    'PROFESSION': 'PROFESSION',

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

NOMBRE_SUJETO_ASISTENCIA: Nombre y apellidos del paciente. Incluye iniciales, apodos o motes.  
NOMBRE_PERSONAL_SANITARIO: Nombre y apellidos de médicos, enfermeros, técnicos u otro personal clínico.  
FAMILIARES_SUJETO_ASISTENCIA: Nombres, apellidos o datos personales de familiares del paciente (edad, parentesco, número).  
ID_SUJETO_ASISTENCIA: Códigos como NHC, CIPA, DNI, NIF, pasaporte u otros identificadores del paciente.  
ID_TITULACIÓN_PERSONAL_SANITARIO: Número de colegiado o licencia del profesional sanitario.  
ID_CONTACTO_ASISTENCIAL: Identificador de episodios clínicos o procesos.  
ID_ASEGURAMIENTO: Número de afiliación a la seguridad social (NASS).  
EDAD_SUJETO_ASISTENCIA: Edad del paciente (incluyendo formas como "tres días", "6 años").  
SEXO_SUJETO_ASISTENCIA: Sexo del paciente (incluyendo formas como "varón", "niña", "M", "H").  
FECHAS: Cualquier fecha del calendario (de nacimiento, ingreso, evolución, etc.).  
CALLE: Dirección postal completa, incluyendo tipo de vía, nombre, número, piso, etc.  
TERRITORIO: Ciudad, provincia, código postal, barrio, comarca, o cualquier otra división geográfica.  
PAÍS: País mencionado en el texto.  
CORREO_ELECTRÓNICO: Cualquier dirección de correo electrónico.  
NÚMERO_TELÉFONO: Números de teléfono personales o profesionales.  
NÚMERO_FAX: Números de fax asociados a la atención o el paciente.  
DIREC_PROT_INTERNET: Direcciones de protocolo de Internet (IP, TCP, SMTP, etc.).  
URL_WEB: Cualquier dirección web o enlace.  
PROFESIÓN: Profesión del paciente o familiares.  
HOSPITAL: Nombres de hospitales o centros sanitarios.  
ID_CENTRO DE SALUD: Nombres de centros de salud o unidades clínicas.  
INSTITUCIÓN: Cualquier otra institución no médica identificable.  
NUMERO_IDENTIF: Otros números de identificación no clasificados.  
IDENTIF_VEHÍCULOS_NRSERIE_PLACAS: Matrículas o números de bastidor de vehículos.  
IDENTIF_DISPOSITIVOS_NRSERIE: Identificadores de dispositivos médicos (serie, chip, etc.).  
IDENTIF_BIOMÉTRICOS: Huellas, escaneos o cualquier identificador biométrico.  
OTROS_SUJETO_ASISTENCIA: Cualquier información adicional que pueda permitir la identificación del paciente y no esté incluida en las categorías anteriores.

🧾 Reglas de anotación estrictas
1. No anotar etiquetas o claves del formulario (como "Nombre:", "Edad:", etc.).
2. No incluir espacios ni signos de puntuación dentro de las etiquetas.
3. Una etiqueta por entidad, aunque se repita en el texto.
4. Etiquetar múltiples palabras como una sola mención si pertenecen a la misma categoría.
5. Excluir títulos o prefijos como "Dr.", "Dña." de las etiquetas de nombres.
6. Anotar todas las fechas, edades, lugares y contactos que puedan identificar al paciente o profesional.
7. Si no se encuentra una entidad entonces no debes incluirla

🧪 Entrada esperada
Cualquier texto clínico en formato libre.

✅ Salida esperada
Devuelve la salida en un **diccionario JSON válido de Python** en una variable llamada annoresult con las siguientes dos claves:

{{
  "texto_anotado": "Texto clínico con etiquetas <<<ETIQUETA>>>...<</ETIQUETA>>> ya insertadas",
  "entidades": {{
    "ETIQUETA1": ["valor1", "valor2", ...],
    "ETIQUETA2": ["valor1", ...]
  }}
}}

El texto que debes analizar es este:
 "{texto_clinico}"
"""

MODEL_NAME = "llama3.2:1B"

# Diccionario de límites de contexto por modelo
MODEL_CONTEXT_LIMITS = {
    "llama3.2:1B": 2048,
    "llama3.3": 2048
}

SAFETY_MARGIN = 0.15  # 15% de margen de seguridad


def check_token_limit(prompt):
    """
    Cuenta tokens con margen de seguridad y muestra métricas detalladas.
    Devuelve True si está dentro del límite seguro.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(prompt)
    total_tokens = len(tokens)
    max_tokens = MODEL_CONTEXT_LIMITS.get(MODEL_NAME, 0)

    # Aplicamos margen de seguridad (15%)
    safe_max_tokens = int(max_tokens * (1 - SAFETY_MARGIN))

    # Informe detallado
    print("\n" + "═" * 50)
    print(f"⚡ Modelo: {MODEL_NAME}")
    print(f"  ► Límite real del modelo: {max_tokens} tokens")
    print(f"  ► Límite seguro ({SAFETY_MARGIN * 100}% margen): {safe_max_tokens} tokens")
    print(f"  ► Tamaño real del prompt: {total_tokens} tokens")
    print(f"  ► Uso del límite real: {total_tokens / max_tokens * 100:.1f}%")
    print("═" * 50)

    if total_tokens > max_tokens:
        print("❌ ¡PELIGRO! Has SUPERADO el LÍMITE REAL del modelo".center(50))
        print("  El prompt será truncado y puede fallar".center(50))
        return False
    elif total_tokens > safe_max_tokens:
        print("⚠️ ¡Atención! Has superado el LÍMITE SEGURO".center(50))
        print("  El prompt funcionará pero podrías tener resultados inesperados".center(50))
        return False
    else:
        print("✅ Dentro del límite seguro".center(50))
        return True


def extract_entities(text):
    """
    Usa el modelo LLM para:
    1. Marcar entidades en el texto con tags <<<TAG>>>
    2. Devolver las entidades en formato clave-valor
    """

    url = "http://localhost:20201/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT_BASE.format(texto_clinico=text),
        "stream": False
    }
    headers = {"Content-Type": "application/json"}

    # Guardar el prompt generado para referencia
    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(payload["prompt"])

    check_token_limit(payload["prompt"])

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=1000)
        response.raise_for_status()

        llm_output = response.json()['response']
        print("Respuesta del LLM:\n", llm_output)

        return {
            'tagged_text': llm_output,
            'entities': {}
        }

    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con el servidor: {e}")
        return {
            'tagged_text': text,
            'entities': {}
        }


def process_xml_files(input_dir, output_dir):
    """Procesa archivos XML manteniendo ambas salidas."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in [f for f in os.listdir(input_dir) if f.endswith('.xml')]:
        try:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"\nProcesando archivo: {filename}")

            # Leer XML original
            tree = ET.parse(input_path)
            original_text = tree.find('.//TEXT').text or ""

            # Extraer datos (texto marcado + entidades)
            result = extract_entities(original_text)

            # Mostrar resultados en consola
            print("Texto marcado:\n", result['tagged_text'])

            # Guardar el resultado en un nuevo XML (simplificado)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['tagged_text'])

            print(f"\nResultado guardado en: {output_path}")
            print("*" * 80)

        except Exception as e:
            print(f"Error procesando {filename}: {str(e)}")


if __name__ == "__main__":
    process_xml_files('test/xml', 'output/xml/quantized_LLaMA_model3.2-1B')