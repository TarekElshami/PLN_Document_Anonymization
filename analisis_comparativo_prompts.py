#!/usr/bin/env python3
"""
Análisis Comparativo de Prompts
Genera visualizaciones comparativas entre dos conjuntos de datos de prompts
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from generate_stats import MeddocanAnalyzer


class ComparativeAnalyzer:
    """Clase para realizar análisis comparativo entre prompts"""

    def __init__(self, output_dir="analisis_comparativo"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')

    def analyze_prompts(self, gold_dir, prompt1_dir, prompt2_dir,
                        prompt1_name="Prompt_1", prompt2_name="Prompt_2"):
        """
        Analiza dos conjuntos de prompts y genera métricas comparativas

        Args:
            gold_dir: Directorio con archivos de referencia
            prompt1_dir: Directorio del primer prompt
            prompt2_dir: Directorio del segundo prompt
            prompt1_name: Nombre del primer prompt
            prompt2_name: Nombre del segundo prompt
        """
        print(f"Comparando {prompt1_name} vs {prompt2_name}...")

        analyzer = MeddocanAnalyzer()

        print(f"Analizando {prompt1_name}...")
        metrics_1 = analyzer.analyze_annotations(gold_dir, prompt1_dir, prompt_name=prompt1_name)

        print(f"Analizando {prompt2_name}...")
        metrics_2 = analyzer.analyze_annotations(gold_dir, prompt2_dir, prompt_name=prompt2_name)

        return self._create_comparison_dataframe(metrics_1, metrics_2, prompt1_name, prompt2_name)

    def _create_comparison_dataframe(self, metrics_1, metrics_2, name1, name2):
        """Crea DataFrame con comparación completa de métricas"""
        comparison_data = []
        all_tags = sorted(list(set(list(metrics_1.keys()) + list(metrics_2.keys()))))
        default_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

        for tag in all_tags:
            m1_metrics = metrics_1.get(tag, default_metrics.copy())
            m2_metrics = metrics_2.get(tag, default_metrics.copy())

            comparison_data.append({
                'Tag': tag,
                f'{name1}_Precision': m1_metrics['precision'],
                f'{name1}_Recall': m1_metrics['recall'],
                f'{name1}_F1': m1_metrics['f1'],
                f'{name2}_Precision': m2_metrics['precision'],
                f'{name2}_Recall': m2_metrics['recall'],
                f'{name2}_F1': m2_metrics['f1'],
                'Diff_Precision': m2_metrics['precision'] - m1_metrics['precision'],
                'Diff_Recall': m2_metrics['recall'] - m1_metrics['recall'],
                'Diff_F1': m2_metrics['f1'] - m1_metrics['f1'],
            })

        df = pd.DataFrame(comparison_data).set_index('Tag')

        # Guardar Excel
        excel_path = self.output_dir / f"comparacion_metricas_{name1}_vs_{name2}.xlsx"
        df.reset_index().to_excel(excel_path, index=False)
        print(f"Tabla de comparación guardada en: {excel_path}")

        return df

    def plot_comparative_heatmap(self, df, metric, prompt1_name, prompt2_name):
        """Genera heatmap comparativo para una métrica específica"""
        cols = [f'{prompt1_name}_{metric}', f'{prompt2_name}_{metric}', f'Diff_{metric}']

        if not all(col in df.columns for col in cols):
            print(f"Advertencia: Faltan columnas para {metric}")
            return

        plt.figure(figsize=(12, max(8, len(df.index) * 0.5)))

        sns.heatmap(df[cols],
                    annot=True,
                    cmap="coolwarm",
                    center=0,
                    fmt=".3f",
                    linewidths=0.5,
                    cbar_kws={'label': f'{metric} Score / Diferencia ({prompt2_name} - {prompt1_name})'})

        plt.title(f'Comparación {metric} ({prompt1_name} vs {prompt2_name}) por Entidad',
                  fontsize=14, fontweight='bold')
        plt.ylabel('Entidad', fontsize=12)
        plt.xlabel('Métricas', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        fig_path = self.output_dir / f"heatmap_{metric.lower()}_{prompt1_name}_vs_{prompt2_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap de {metric} guardado en: {fig_path}")
        plt.show()

    def generate_all_heatmaps(self, df, prompt1_name, prompt2_name):
        """Genera todos los heatmaps comparativos"""
        metrics = ['F1', 'Precision', 'Recall']

        for metric in metrics:
            self.plot_comparative_heatmap(df, metric, prompt1_name, prompt2_name)

    def analyze_text_lengths(self, folder1, folder2, name1="Prompt1", name2="Prompt2"):
        """
        Analiza y compara las longitudes de texto entre dos carpetas

        Args:
            folder1: Ruta de la primera carpeta
            folder2: Ruta de la segunda carpeta
            name1: Nombre del primer conjunto
            name2: Nombre del segundo conjunto
        """
        print(f"Analizando longitudes de texto: {name1} vs {name2}")

        # Obtener archivos comunes
        files1 = set(f for f in os.listdir(folder1) if f.endswith('.txt'))
        files2 = set(f for f in os.listdir(folder2) if f.endswith('.txt'))
        common_files = files1 & files2

        if not common_files:
            print("No se encontraron archivos comunes para comparar")
            return

        # Analizar diferencias
        longer_in_1, longer_in_2, equal_length = 0, 0, 0
        file_differences = []

        for file in common_files:
            with open(os.path.join(folder1, file), 'r', encoding='utf-8') as f1, \
                    open(os.path.join(folder2, file), 'r', encoding='utf-8') as f2:
                len1 = len(f1.read())
                len2 = len(f2.read())
                difference = len1 - len2

                if len1 > len2:
                    longer_in_1 += 1
                elif len2 > len1:
                    longer_in_2 += 1
                else:
                    equal_length += 1

                file_differences.append((file, difference, len1, len2))

        # Generar visualizaciones
        self._plot_length_comparison(longer_in_1, longer_in_2, equal_length,
                                     len(common_files), name1, name2)
        self._plot_length_differences(file_differences, name1, name2)
        self._print_length_statistics(file_differences, longer_in_1, longer_in_2,
                                      equal_length, len(common_files), name1, name2)

    def _plot_length_comparison(self, longer_1, longer_2, equal, total, name1, name2):
        """Gráfico de barras con porcentajes de longitud"""
        percentages = [(longer_1 / total) * 100, (longer_2 / total) * 100, (equal / total) * 100]
        labels = [f'{name1} más largo', f'{name2} más largo', 'Igual longitud']

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, percentages, color=['#3498db', '#e74c3c', '#2ecc71'])
        plt.title(f'Comparación de longitud entre {name1} y {name2}',
                  fontsize=14, fontweight='bold')
        plt.ylabel('Porcentaje de archivos (%)', fontsize=12)
        plt.ylim(0, 100)

        # Añadir etiquetas en las barras
        for bar, pct in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{pct:.1f}%", ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        fig_path = self.output_dir / f"comparacion_longitudes_{name1}_vs_{name2}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de comparación guardado en: {fig_path}")
        plt.show()

    def _plot_length_differences(self, file_differences, name1, name2):
        """Gráfico de dispersión de diferencias de longitud"""
        differences = [diff for (_, diff, _, _) in file_differences]

        plt.figure(figsize=(12, 6))
        scatter = plt.scatter(range(len(differences)), differences, alpha=0.6, s=50)
        plt.axhline(0, color='red', linestyle='--', alpha=0.7)
        plt.title(f'Diferencia de longitud por archivo ({name1} - {name2})',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Archivo (índice)', fontsize=12)
        plt.ylabel('Diferencia de longitud (caracteres)', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Añadir estadísticas en el gráfico
        mean_diff = np.mean(differences)
        plt.text(0.02, 0.98, f'Diferencia promedio: {mean_diff:.1f}',
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        fig_path = self.output_dir / f"diferencias_longitud_{name1}_vs_{name2}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de diferencias guardado en: {fig_path}")
        plt.show()

    def _print_length_statistics(self, file_differences, longer_1, longer_2,
                                 equal, total, name1, name2):
        """Imprime estadísticas detalladas"""
        differences = [diff for (_, diff, _, _) in file_differences]

        print(f"\n{'=' * 60}")
        print(f"ESTADÍSTICAS DE COMPARACIÓN: {name1} vs {name2}")
        print(f"{'=' * 60}")
        print(f"Total archivos comparados: {total}")
        print(f"{name1} más largo: {longer_1} veces ({(longer_1 / total) * 100:.1f}%)")
        print(f"{name2} más largo: {longer_2} veces ({(longer_2 / total) * 100:.1f}%)")
        print(f"Igual longitud: {equal} veces ({(equal / total) * 100:.1f}%)")

        if differences:
            avg_diff = np.mean(differences)
            print(f"Diferencia promedio: {avg_diff:.1f} caracteres")
            print(f"  (positivo = {name1} más largo)")

        # Top 5 diferencias más grandes
        sorted_diffs = sorted(file_differences, key=lambda x: abs(x[1]), reverse=True)

        print(f"\nTOP 5 MAYORES DIFERENCIAS:")
        print("-" * 60)
        for file, diff, len1, len2 in sorted_diffs[:5]:
            tendency = f"{name1}+" if diff > 0 else f"{name2}+"
            print(f"{file:<30} | {diff:>8} | {tendency}")