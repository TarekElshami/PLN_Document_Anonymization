import os
import subprocess
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import tempfile


class EvaluationSystem:
    """Sistema de evaluación para modelos de NER"""

    def __init__(self, results_dir="results", system_wrong="systemWrong"):
        self.results_dir = results_dir
        self.system_wrong = system_wrong
        self.stats_file_template = os.path.join(results_dir, "stats__{}.csv")
        self.graph_file_template = os.path.join(results_dir, "metrics__{}__{}.png")
        os.makedirs(results_dir, exist_ok=True)

    def run_evaluation(self, test_dir, system_dir, eval_type="ner"):
        """Ejecuta la evaluación usando evaluate.py"""
        cmd = ["python", "evaluate.py", "i2b2", eval_type, test_dir, system_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout

    def sort_by_prompt_number(self, lst):
        """Ordena lista por número de prompt extraído del nombre"""

        def extract_number(s):
            match = re.search(r'(\d+)', s)
            return int(match.group(1)) if match else float('inf')

        return sorted(lst, key=extract_number)

    def sanitize_filename(self, name):
        """Limpia nombre de archivo para uso seguro"""
        return name.strip("/").replace("/", "_")

    def parse_evaluation_output(self, output):
        """Extrae métricas del output de evaluación"""
        stats = {"Leak": 0, "Precision": 0, "Recall": 0, "F1": 0}
        for line in output.splitlines():
            if "Leak" in line and "Leakage" not in line:
                leak = re.search(r"Leak\s+([\d.]+)", line)
                if leak:
                    stats["Leak"] = float(leak.group(1))
            elif "Precision" in line:
                prec = re.search(r"Precision\s+([\d.]+)", line)
                if prec:
                    stats["Precision"] = float(prec.group(1))
            elif "Recall" in line:
                rec = re.search(r"Recall\s+([\d.]+)", line)
                if rec:
                    stats["Recall"] = float(rec.group(1))
            elif "F1" in line:
                f1 = re.search(r"F1\s+([\d.]+)", line)
                if f1:
                    stats["F1"] = float(f1.group(1))
        return stats

    def get_xml_files(self, directory):
        """Obtiene conjunto de archivos XML en directorio"""
        return {f for f in os.listdir(directory) if f.endswith(".xml")}

    def prepare_mixed_config(self, config_dir, expected_files):
        """Prepara directorio temporal con archivos de configuración y systemWrong"""
        mixed_dir = tempfile.mkdtemp()
        missing_files = set(expected_files)

        # Copiar archivos del sistema principal
        for root, _, files in os.walk(config_dir):
            for file in files:
                if file in expected_files:
                    shutil.copy2(os.path.join(root, file), os.path.join(mixed_dir, file))
                    missing_files.discard(file)

        # Añadir archivos faltantes de systemWrong
        if missing_files and os.path.exists(self.system_wrong):
            print(f"    🔄 Usando {len(missing_files)} archivos de systemWrong")
            for root, _, files in os.walk(self.system_wrong):
                for file in files:
                    if file in missing_files:
                        shutil.copy2(os.path.join(root, file), os.path.join(mixed_dir, file))

        return mixed_dir

    def save_stats(self, all_stats, system_name):
        """Guarda estadísticas en CSV"""
        filepath = self.stats_file_template.format(system_name)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Config", "Leak", "Precision", "Recall", "F1"])
            writer.writeheader()
            for config, metrics in all_stats.items():
                row = {"Config": config, **metrics}
                writer.writerow(row)
        print(f"📊 Stats guardadas en: {filepath}")

    def generate_single_graph(self, df, system_name, eval_type="ner"):
        """Genera gráfica individual para un sistema y tipo de evaluación"""
        plt.figure(figsize=(14, 7))

        # Solo mostrar Leak si no es evaluación de spans
        if eval_type != "spans":
            plt.plot(df["Config"], df["Leak"], label="Leak", marker="o", linestyle="--", color="red")

        plt.plot(df["Config"], df["Precision"], label="Precision", marker="s", color="blue")
        plt.plot(df["Config"], df["Recall"], label="Recall", marker="^", color="green")
        plt.plot(df["Config"], df["F1"], label="F1", marker="D", color="purple")

        plt.title(f"Métricas ({eval_type}) - {system_name}", fontsize=14)
        plt.xlabel("Configuración", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        graph_path = self.graph_file_template.format(system_name, eval_type)
        plt.savefig(graph_path)
        print(f"✅ Gráfica guardada en: {graph_path}")
        plt.show()

    def generate_comparison_graph(self, df_ner, df_spans, system_name):
        """Genera gráfica comparativa entre NER y Spans"""
        plt.figure(figsize=(14, 7))

        for metric in ["Precision", "Recall", "F1"]:
            plt.plot(df_ner["Config"], df_ner[metric],
                     label=f"{metric} (NER)", linestyle="--", marker="o")
            plt.plot(df_spans["Config"], df_spans[metric],
                     label=f"{metric} (Spans)", linestyle="-", marker="s")

        plt.title(f"Comparativa NER vs Spans - {system_name}", fontsize=14)
        plt.xlabel("Configuración", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        graph_path = self.graph_file_template.format(system_name, "comparativa")
        plt.savefig(graph_path)
        print(f"✅ Gráfica comparativa guardada en: {graph_path}")
        plt.show()

    def evaluate_system(self, system_path, test_dir, eval_types=["ner"]):
        """Evalúa un sistema completo con todos sus configs"""
        if not os.path.exists(system_path):
            print(f"⚠️  Sistema no encontrado: {system_path}")
            return {}

        expected_files = self.get_xml_files(test_dir)
        if not expected_files:
            print("❌ Error: No hay archivos XML en test/xml/")
            return {}

        print(f"\n📊 Procesando: {system_path}")
        configs = self.sort_by_prompt_number(os.listdir(system_path))

        results = {}
        for eval_type in eval_types:
            print(f"\n🔧 Evaluación: {eval_type}")
            all_stats = {}

            for config in configs:
                config_dir = os.path.join(system_path, config)
                if not os.path.isdir(config_dir):
                    continue

                print(f"  ⚙️  Config: {config}")
                mixed_dir = self.prepare_mixed_config(config_dir, expected_files)

                try:
                    output = self.run_evaluation(test_dir, mixed_dir, eval_type)
                    stats = self.parse_evaluation_output(output)
                    all_stats[config] = stats
                    print(f"    ✅ Leak: {stats['Leak']:.3f} | F1: {stats['F1']:.3f} | "
                          f"P: {stats['Precision']:.3f} | R: {stats['Recall']:.3f}")
                finally:
                    shutil.rmtree(mixed_dir)

            if all_stats:
                safe_name = self.sanitize_filename(system_path)
                df = pd.DataFrame.from_dict(all_stats, orient="index").reset_index()
                df = df.rename(columns={"index": "Config"})

                # Guardar stats solo para NER (por compatibilidad)
                if eval_type == "ner":
                    self.save_stats(all_stats, safe_name)

                # Generar gráfica
                self.generate_single_graph(df, safe_name, eval_type)
                results[eval_type] = df

        return results

    def run_full_evaluation(self, systems, test_dir="test/xml/", eval_types=["ner"]):
        """Ejecuta evaluación completa para múltiples sistemas"""
        if not os.path.exists(test_dir):
            print(f"❌ Error: No existe {test_dir}")
            return

        expected_files = self.get_xml_files(test_dir)
        print(f"\n🔍 Archivos de referencia: {len(expected_files)} en {test_dir}")

        all_results = {}
        for system_path in systems:
            system_results = self.evaluate_system(system_path, test_dir, eval_types)

            # Generar gráfica comparativa si hay NER y Spans
            if "ner" in system_results and "spans" in system_results:
                safe_name = self.sanitize_filename(system_path)
                self.generate_comparison_graph(
                    system_results["ner"],
                    system_results["spans"],
                    safe_name
                )

            all_results[system_path] = system_results

        return all_results


# Función de conveniencia para uso rápido
def quick_evaluation(systems, eval_types=["ner"], results_dir="results"):
    """Función de acceso rápido para evaluación"""
    evaluator = EvaluationSystem(results_dir=results_dir)
    return evaluator.run_full_evaluation(systems, eval_types=eval_types)