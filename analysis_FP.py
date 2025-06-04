import re
import sys
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import unicodedata

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
    return texto.strip()

def extraer_fps_y_extras(bloque):
    fps = []
    tag_actual = None

    lineas = bloque.strip().splitlines()
    i = 0
    while i < len(lineas):
        linea = lineas[i].strip()

        match_fp = re.match(r"✗ \[\d+-\d+\] '(.*?)' \((.*?)\)", linea)
        if match_fp:
            texto_fp = match_fp.group(1)
            tag_actual = match_fp.group(2)
            fps.append((texto_fp, tag_actual))

        # Buscar contenidos extra (indentados)
        if 'Contenido significativo extra' in linea:
            i += 1
            while i < len(lineas) and '-' in lineas[i]:
                match_extra = re.search(r"-\s*'(.*?)'", lineas[i])
                if match_extra:
                    extra = match_extra.group(1)
                    if tag_actual:
                        fps.append((extra, tag_actual))
                i += 1
            continue

        i += 1

    return fps

def procesar_txt(ruta):
    with open(ruta, encoding='utf-8') as f:
        contenido = f.read()

    # Extraer secciones de FP (los dos bloques relevantes)
    secciones_fp = re.findall(
        r"(FALSOS POSITIVOS(?: ADICIONALES POR CONTENIDO EXTRA| NO EMPAREJADOS).*?):\n(.*?)(?=\n[A-Z]|$)",
        contenido,
        re.DOTALL
    )

    conteo_fp_texto = Counter()
    conteo_tags = defaultdict(int)

    for nombre, bloque in secciones_fp:
        fps = extraer_fps_y_extras(bloque)
        for texto, tag in fps:
            texto_limpio = limpiar_texto(texto)
            conteo_fp_texto[texto_limpio] += 1
            conteo_tags[tag] += 1

    # Mostrar top 15 FPs más repetidos
    print("\nTop 15 Falsos Positivos más repetidos (normalizados):\n")
    for texto, count in conteo_fp_texto.most_common(15):
        print(f"{texto}: {count}")

    # Gráfica de los tags más frecuentes
    if conteo_tags:
        etiquetas, valores = zip(*sorted(conteo_tags.items(), key=lambda x: x[1], reverse=True))
        plt.figure(figsize=(12, 6))
        plt.bar(etiquetas, valores, color='orange')
        plt.xticks(rotation=45, ha='right')
        plt.title('Tags con más Falsos Positivos')
        plt.ylabel('Cantidad')
        plt.tight_layout()
        plt.show()