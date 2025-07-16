import os
import json
import re


def convert_to_bio(tagged_text):
    lines = []
    # Patrón para encontrar etiquetas y su contenido
    pattern = re.compile(r'<([A-Z]+)>(.*?)</\1>')

    # Encontrar todas las etiquetas y sus posiciones
    tags = []
    for match in pattern.finditer(tagged_text):
        tag = match.group(1)
        content = match.group(2)
        start, end = match.span()
        tags.append((start, end, tag, content))

    # Procesar el texto por partes
    last_end = 0
    for start, end, tag, content in tags:
        # Procesar texto antes de la etiqueta (O)
        before_text = tagged_text[last_end:start].strip()
        if before_text:
            for word in before_text.split():
                lines.append(f"{word} O")

        # Procesar el contenido etiquetado (B-TAG, I-TAG...)
        words = content.split()
        for i, word in enumerate(words):
            if i == 0:
                lines.append(f"{word} B-{tag}")
            else:
                lines.append(f"{word} I-{tag}")

        last_end = end

    # Procesar texto después de la última etiqueta (O)
    after_text = tagged_text[last_end:].strip()
    if after_text:
        for word in after_text.split():
            lines.append(f"{word} O")

    return '\n'.join(lines)


def process_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('_llm_response.txt'):
            input_path = os.path.join(input_dir, filename)
            output_filename = filename.replace('_llm_response.txt', '.bio')
            output_path = os.path.join(output_dir, output_filename)

            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tagged_text = data['tagged_text']
                bio_text = convert_to_bio(tagged_text)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(bio_text)


# Example usage
input_directory = "systemLlama3.3/promptCoNLL/"
output_directory = "hola/"
process_files(input_directory, output_directory)