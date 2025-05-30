import os
from pathlib import Path
from custom_evaluator import evaluate_with_margin
from classes import BratAnnotation, i2b2Annotation

FORMAT = "i2b2"
GS_DIR = "test/xml"
SYS_DIR = ["procesados_meddocan/systemLlama3.3/etiquetado/prompt8/"]
BASE_OUTPUT_DIR = "errores"

# Crear estructura de directorios
Path(BASE_OUTPUT_DIR).mkdir(exist_ok=True)

# Ejecutar evaluaciones para todos los márgenes
for margin in [i / 100 for i in range(0, 105, 5)]:
    for use_tags in [True, False]:
        # Configurar nombre de subcarpeta
        tag_status = "with_tags" if use_tags else "no_tags"
        margin_str = f"{margin:.2f}".replace(".", "_")
        output_dir = os.path.join(BASE_OUTPUT_DIR, tag_status, f"margin_{margin_str}")

        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nEvaluando: margin={margin:.2f}, tags={'ON' if use_tags else 'OFF'}")

        # Seleccionar la clase de anotación según el formato
        annotation_class = BratAnnotation if FORMAT == "brat" else i2b2Annotation

        # Ejecutar evaluación directamente
        evaluate_with_margin(
            gs_path=GS_DIR,
            sys_paths=SYS_DIR,
            annotation_class=annotation_class,
            margin=margin,
            ignore_tags=not use_tags,
            output_dir=output_dir,
            verbose=False
        )


print("\n¡Todas las evaluaciones completadas!")