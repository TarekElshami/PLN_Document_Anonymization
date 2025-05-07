import os
import subprocess
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import tempfile

# Configuración
RESULTS_DIR = "results"
MAIN_SYSTEMS = ["systemLlama3.3", "systemLlamaQuantized"]
SYSTEM_WRONG = "systemWrong"
STATS_FILE = os.path.join(RESULTS_DIR, "stats_{}.csv")
GRAPH_FILE = os.path.join(RESULTS_DIR, "metrics_{}.png")  # Solo una gráfica


def run_evaluation(test_dir, system_dir):
    cmd = ["python", "evaluate.py", "i2b2", "ner", test_dir, system_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def parse_evaluation_output(output):
    stats = {"Leak": 0, "Precision": 0, "Recall": 0, "F1": 0}  # Valores por defecto
    for line in output.splitlines():
        if "Leak" in line and not "Leakage" in line:
            leak = re.search(r"Leak\s+([\d.]+)", line)
            if leak: stats["Leak"] = float(leak.group(1))
        elif "Precision" in line:
            prec = re.search(r"Precision\s+([\d.]+)", line)
            if prec: stats["Precision"] = float(prec.group(1))
        elif "Recall" in line:
            rec = re.search(r"Recall\s+([\d.]+)", line)
            if rec: stats["Recall"] = float(rec.group(1))
        elif "F1" in line:
            f1 = re.search(r"F1\s+([\d.]+)", line)
            if f1: stats["F1"] = float(f1.group(1))
    return stats


def save_stats(all_stats, system_name):
    with open(STATS_FILE.format(system_name), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Config", "Leak", "Precision", "Recall", "F1"])
        writer.writeheader()
        for config, metrics in all_stats.items():
            row = {"Config": config, **metrics}
            writer.writerow(row)


def generar_grafica_unica(df, system_name):
    plt.figure(figsize=(14, 7))
    plt.plot(df["Config"], df["Leak"], label="Leak", marker="o", linestyle="--", color="red")
    plt.plot(df["Config"], df["Precision"], label="Precision", marker="s", color="blue")
    plt.plot(df["Config"], df["Recall"], label="Recall", marker="^", color="green")
    plt.plot(df["Config"], df["F1"], label="F1", marker="D", color="purple")

    plt.title(f"Métricas - {system_name}", fontsize=14)
    plt.xlabel("Configuración", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(GRAPH_FILE.format(system_name))
    plt.close()
    print(f"✅ Gráfica guardada en: {GRAPH_FILE.format(system_name)}")


def get_xml_files(directory):
    return {f for f in os.listdir(directory) if f.endswith(".xml")}


def prepare_mixed_config(config_dir, expected_files):
    mixed_dir = tempfile.mkdtemp()
    missing_files = set(expected_files)

    # Copiar archivos del sistema principal
    for root, _, files in os.walk(config_dir):
        for file in files:
            if file in expected_files:
                shutil.copy2(os.path.join(root, file), os.path.join(mixed_dir, file))
                missing_files.remove(file)

    # Añadir archivos faltantes de systemWrong (si existen)
    if missing_files and os.path.exists(SYSTEM_WRONG):
        print(f"    🔄 Usando {len(missing_files)} archivos de systemWrong")
        for root, _, files in os.walk(SYSTEM_WRONG):
            for file in files:
                if file in missing_files:
                    shutil.copy2(os.path.join(root, file), os.path.join(mixed_dir, file))

    return mixed_dir


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Verificar archivos de test
    test_dir = "test/xml/"
    if not os.path.exists(test_dir):
        print(f"❌ Error: No existe {test_dir}")
        return

    expected_files = get_xml_files(test_dir)
    if not expected_files:
        print("❌ Error: No hay archivos XML en test/xml/")
        return

    print(f"\n🔍 Archivos de referencia: {len(expected_files)} en test/xml/")

    for system_name in MAIN_SYSTEMS:
        system_path = system_name
        if not os.path.exists(system_path):
            print(f"\n⚠️  Sistema no encontrado: {system_path}")
            continue

        print(f"\n📊 Procesando: {system_name}")
        all_stats = {}

        for config in sorted(os.listdir(system_path)):
            config_dir = os.path.join(system_path, config)
            if not os.path.isdir(config_dir):
                continue

            print(f"  🔧 Config: {config}")
            mixed_dir = prepare_mixed_config(config_dir, expected_files)

            try:
                output = run_evaluation(test_dir, mixed_dir)
                stats = parse_evaluation_output(output)
                all_stats[config] = stats
                print(
                    f"    ✅ Leak: {stats['Leak']:.3f} | F1: {stats['F1']:.3f} | P: {stats['Precision']:.3f} | R: {stats['Recall']:.3f}")
            finally:
                shutil.rmtree(mixed_dir)

        if all_stats:
            save_stats(all_stats, system_name)
            df = pd.DataFrame.from_dict(all_stats, orient="index").reset_index().rename(columns={"index": "Config"})
            generar_grafica_unica(df, system_name)


if __name__ == "__main__":
    main()