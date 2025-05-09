import os
import json
import logging
from tqdm import tqdm

from anonymizer import process_single_xml_file

# --- Configuración ---
INPUT_DIR = "test/xml/"
PROMPTS_DIR = "prompts"
OLLAMA_PORT = "20201"
STATE_FILE = "processed_state.json"

# Modelos a procesar y sus directorios de salida
MODELS = {
    'llama3.3': 'systemLlama3.3',
    'llama3.2:1B': 'systemLlamaQuantized'
}

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_anonymizer.log'),
        # logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    prompt_files = [f for f in os.listdir(PROMPTS_DIR) if f.endswith('.txt') and f != 'prompt1.txt']

    for prompt_file in prompt_files:
        with open(os.path.join(PROMPTS_DIR, prompt_file), 'r', encoding='utf-8') as f:
            prompt_text = f.read()
        prompt_name = os.path.splitext(prompt_file)[0]

        for model_name, model_dir in MODELS.items():
            output_dir = os.path.join(model_dir, prompt_name)
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Procesando con modelo '{model_name}' y prompt '{prompt_name}'")
            for input_file in tqdm(input_files, desc=f"{model_name}/{prompt_name}"):
                input_path = os.path.join(INPUT_DIR, input_file)

                key = f"{model_name}/{prompt_name}/{input_file}"
                if key in state:
                    continue  # Ya procesado

                success = process_single_xml_file(
                    input_file_path=input_path,
                    output_dir=output_dir,
                    ollama_port=OLLAMA_PORT,
                    model_name=model_name,
                    prompt_text=prompt_text
                )

                if success:
                    state[key] = True
                    save_processed_files(state)
                else:
                    logger.error(f"Falló el procesamiento de {input_file} con modelo {model_name} y prompt {prompt_name}")

# --- Punto de entrada ---
if __name__ == "__main__":
    process_all_prompts()
