import os
import subprocess
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt

PROMPTS_DIR = "prompts"
RESULTS_DIR = "results"
SYSTEM_DIR = "system"
STATS_FILE = "stats.csv"

def run_evaluation(prompt_name):
    system_path = os.path.join(SYSTEM_DIR, prompt_name)
    cmd = ["python", "evaluate.py", "i2b2", "ner", "test/xml/", system_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def parse_evaluation_output(output):
    stats = {"Leak": None, "Precision": None, "Recall": None, "F1": None}
    for line in output.splitlines():
        # Buscar lineas que contengan las métricas
        if "Leak" in line and "Leakage" not in line:
            match = re.search(r"Leak\s+([\d.]+)", line)
            if match:
                stats["Leak"] = float(match.group(1))
        elif "Precision" in line:
            match = re.search(r"Precision\s+([\d.]+)", line)
            if match:
                stats["Precision"] = float(match.group(1))
        elif "Recall" in line:
            match = re.search(r"Recall\s+([\d.]+)", line)
            if match:
                stats["Recall"] = float(match.group(1))
        elif "F1" in line:
            match = re.search(r"F1\s+([\d.]+)", line)
            if match:
                stats["F1"] = float(match.group(1))
    return stats


def save_stats(all_stats):
    with open(STATS_FILE, "w", newline='') as csvfile:
        fieldnames = ["Prompt", "Leak", "Precision", "Recall", "F1"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_stats:
            writer.writerow(row)


def generar_graficas():
    df = pd.read_csv(STATS_FILE)

    # Ordenar por nombre de prompt
    df = df.sort_values("Prompt")

    # Crear gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(df["Prompt"], df["Leak"], label="Leak", marker='o')
    plt.plot(df["Prompt"], df["Precision"], label="Precision", marker='s')
    plt.plot(df["Prompt"], df["Recall"], label="Recall", marker='^')
    plt.plot(df["Prompt"], df["F1"], label="F1 Score", marker='x')
    plt.xlabel("Prompt")
    plt.ylabel("Score")
    plt.title("Evolución de Leak, Precision, Recall y F1 Score")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("metrics_over_time.png")
    plt.close()

    print("Gráfica generada: metrics_over_time.png")

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    prompts = [f for f in os.listdir(PROMPTS_DIR) if f.endswith('.txt')]
    prompts = sorted(prompts)  # Orden alfabético: prompt001.txt, prompt002.txt, etc

    all_stats = []

    for prompt_file in prompts:
        prompt_name = os.path.splitext(prompt_file)[0]
        print(f"Evaluando {prompt_name}...")

        # Ejecutar evaluación
        output = run_evaluation(prompt_name)

        # Guardar output
        prompt_result_dir = os.path.join(RESULTS_DIR, prompt_name)
        os.makedirs(prompt_result_dir, exist_ok=True)
        with open(os.path.join(prompt_result_dir, "eval_output.txt"), "w") as f:
            f.write(output)

        # Parsear y guardar estadísticas
        stats = parse_evaluation_output(output)
        stats["Prompt"] = prompt_name
        all_stats.append(stats)

    # Guardar stats
    save_stats(all_stats)

    # Generar gráficas
    generar_graficas()

if __name__ == "__main__":
    main()
