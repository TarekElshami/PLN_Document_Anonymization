import os
import xml.etree.ElementTree as ET
from collections import defaultdict


def analyze_xml_tags(folder_path):
    tag_type_counts = defaultdict(lambda: defaultdict(int))
    tag_totals = defaultdict(int)     # Total por categoría (tag)
    type_totals = defaultdict(int)    # Total por entidad (type)
    total_global = 0

    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder_path, filename)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                tags_section = root.find('TAGS')
                if tags_section is not None:
                    for tag in tags_section:
                        tag_name = tag.tag
                        tag_type = tag.get('TYPE', 'SIN_TIPO')

                        tag_type_counts[tag_name][tag_type] += 1
                        tag_totals[tag_name] += 1
                        type_totals[tag_type] += 1
                        total_global += 1

            except ET.ParseError as e:
                print(f"Error al parsear {filename}: {e}")
            except Exception as e:
                print(f"Error procesando {filename}: {e}")

    # Mostrar resultados
    print(f"Total global de entidades: {total_global}\n")
    print("Análisis de tags y tipos con porcentaje global:\n")

    for tag, type_counts in sorted(tag_type_counts.items()):
        total_tag = tag_totals[tag]
        pct_tag = (total_tag / total_global) * 100 if total_global else 0
        print(f"[{tag}] - Total categoría: {total_tag} ({pct_tag:.2f}%)")
        for type_name, count in sorted(type_counts.items()):
            pct_type = (count / total_global) * 100 if total_global else 0
            print(f"  {type_name}: {count} ({pct_type:.2f}%)")

    print("\nTotales por entidad (TYPE):")
    for type_name, count in sorted(type_totals.items(), key=lambda x: -x[1]):
        pct = (count / total_global) * 100 if total_global else 0
        print(f"  {type_name}: {count} ({pct:.2f}%)")


if __name__ == "__main__":
    folder_path = input("Introduce la ruta de la carpeta con los archivos XML: ")
    if os.path.isdir(folder_path):
        analyze_xml_tags(folder_path)
    else:
        print("La ruta especificada no es una carpeta válida.")
