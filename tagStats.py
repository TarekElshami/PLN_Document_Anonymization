import re
from pathlib import Path

def calculate_tag_quality(directory, pattern_num=1):
    # Seleccionar el patrón correcto según el número
    patterns = [
        re.compile(r'(?<!<)<<<([A-Z_]+)>>>.*?(?<!<)<<</\1>>>', re.DOTALL),  # Pattern 1
        re.compile(r'(?<!<)<(?!<)([A-Z_]+)>.*?(?<!<)</\1>', re.DOTALL),      # Pattern 2
        re.compile(r'(?<!<)<\*(?!\*)([A-Z_]+)>.*?(?<!<)<\*/\1>', re.DOTALL), # Pattern 3
        re.compile(r'(?<!<)<\*(?!\*)([A-Z_]+)\*>.*?(?<!<)<\*/\1\*>', re.DOTALL) # Pattern 4
    ]
    correct_pattern = patterns[pattern_num - 1]

    lt_group_pattern = re.compile(r'<+')

    correct_count = 0
    wrong_count = 0
    errors = []  # Lista para almacenar los errores encontrados

    for txt_file in Path(directory).glob('*.txt'):
        try:
            content = txt_file.read_text(encoding='utf-8')

            # 1. Encontrar etiquetas correctas
            correct_matches = list(correct_pattern.finditer(content))
            correct_count += len(correct_matches)

            # 2. Buscar todos los grupos de <
            for lt_match in lt_group_pattern.finditer(content):
                lt_group = lt_match.group()

                # Verificar si es parte de una etiqueta válida
                is_valid = False
                for valid_match in correct_matches:
                    if (lt_match.start() >= valid_match.start() and
                            lt_match.end() <= valid_match.end()):
                        is_valid = True
                        break

                if not is_valid:
                    wrong_count += 1
                    # Obtener contexto del error
                    start = max(0, lt_match.start() - 20)
                    end = min(len(content), lt_match.end() + 20)
                    context = content[start:end].replace('\n', ' ')
                    errors.append({
                        'file': str(txt_file),
                        'position': (lt_match.start(), lt_match.end()),
                        'error': lt_group,
                        'context': f"...{context}..."
                    })

        except Exception as e:
            print(f"[!] Error al procesar {txt_file}: {e}")
            continue

    # Calcular métrica
    adjusted_wrong = wrong_count / 2
    total = correct_count + adjusted_wrong
    quality = correct_count / total if total > 0 else 1.0

    return {
        'correct_tags': correct_count,
        'wrong_groups': wrong_count,
        'adjusted_wrong': adjusted_wrong,
        'quality_metric': quality,
        'total_errors': len(errors)
    }


if __name__ == "__main__":
    results = calculate_tag_quality("systemLlama3.3/prompt3/")

    print("\n=== RESUMEN ===")
    print(f"✔ Etiquetas correctas: {results['correct_tags']}")
    print(f"✖ Grupos < mal formados: {results['wrong_groups']}")
    print(f"✖ Ajustado (÷2): {results['adjusted_wrong']}")
    print(f"📊 Métrica de calidad: {results['quality_metric']:.2%}")
    print(f"🚫 Total de errores mostrados: {results['total_errors']}")