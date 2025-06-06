import os
import json
import logging
import configparser
from tqdm import tqdm
import llm_client

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logBatch.log', mode='w')
        # logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Cargar configuración ---
config = configparser.ConfigParser(delimiters=('=',))
config.optionxform = str
config.read('batchConfig.ini')

# --- Configuración ---
INPUT_DIR = config.get('General', 'input_dir')
PROMPTS_DIR = config.get('General', 'prompts_dir')
OLLAMA_PORT = config.get('General', 'ollama_port')
STATE_FILE = config.get('General', 'state_file')

# Límite opcional para pruebas
try:
    max_files_raw = config.get('General', 'max_files', fallback=None)
    if max_files_raw is None or max_files_raw.strip() == '' or int(max_files_raw) <= 0:
        MAX_FILES = None  # Sin límite
    else:
        MAX_FILES = int(max_files_raw)
except (ValueError, configparser.NoOptionError) as e:
    logger.warning(f"No se pudo interpretar 'max_files': {e}")
    MAX_FILES = None

# Modelos a procesar y sus directorios de salida
MODELS = {key: value for key, value in config.items('Models')}

# Diccionario de límites de contexto por modelo
MODEL_CONTEXT_LIMITS = {}
try:
    for key, value in config.items('ModelContextLimits'):
        try:
            MODEL_CONTEXT_LIMITS[key] = int(value)
            if MODEL_CONTEXT_LIMITS[key] <= 0:
                raise ValueError(f"El límite de contexto para '{key}' debe ser mayor que 0")
        except ValueError as e:
            logger.error(f"Error en la configuración de límites de contexto: {e}")
            raise SystemExit("Error fatal en la configuración. Deteniendo la ejecución.")
except configparser.NoSectionError:
    logger.error("Sección 'ModelContextLimits' no encontrada en batchConfig.ini")
    raise SystemExit("Error fatal: Sección requerida no encontrada. Deteniendo la ejecución.")

# --- Estado de procesamiento ---
def load_processed_files():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_processed_files(state):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


# --- Procesamiento Principal ---
def process_all_prompts():
    state = load_processed_files()
    input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.xml')]
    if MAX_FILES is not None:
        input_files = input_files[:MAX_FILES]

    # Prompts a procesar: prompt8 y prompt11
    prompt_files = ['prompt8.txt', 'prompt11.txt']

    # Verificar que los archivos de prompt existen
    missing_prompts = []
    for prompt_file in prompt_files:
        if prompt_file not in os.listdir(PROMPTS_DIR):
            missing_prompts.append(prompt_file)

    if missing_prompts:
        logger.error(f"Error: Los siguientes archivos de prompt no se encontraron: {missing_prompts}")
        return

    for prompt_file in prompt_files:
        with open(os.path.join(PROMPTS_DIR, prompt_file), 'r', encoding='utf-8') as f:
            full_text = f.read()
            prompt_text = full_text.split("Tarea:", 1)[-1].strip()

        prompt_name = os.path.splitext(prompt_file)[0]

        # Determinar si debe usarse la gramática
        use_grammar = not prompt_file == 'prompt1.txt'

        for model_name, model_dir in MODELS.items():
            output_dir = os.path.join(model_dir, prompt_name)
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Procesando con modelo '{model_name}' y prompt '{prompt_name}'")
            for input_file in tqdm(input_files, desc=f"{model_name}/{prompt_name}"):
                input_path = os.path.join(INPUT_DIR, input_file)

                key = f"{model_name}/{prompt_name}/{input_file}"
                if key in state:
                    continue  # Ya procesado

                success = llm_client.process_llm_request(
                    input_file_path=input_path,
                    output_dir=output_dir,
                    ollama_port=OLLAMA_PORT,
                    model_name=model_name,
                    prompt_text=prompt_text,
                    use_grammar=use_grammar,
                    context_size=MODEL_CONTEXT_LIMITS.get(model_name)
                )
                if success:
                    state[key] = True
                    save_processed_files(state)
                else:
                    logger.error(
                        f"Falló el procesamiento de {input_file} con modelo {model_name} y prompt {prompt_name}")


# --- Punto de entrada ---
if __name__ == "__main__":
    process_all_prompts()