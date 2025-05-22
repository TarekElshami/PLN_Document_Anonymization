import os
from collections import defaultdict
from lxml import etree
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MeddocanAnalyzer:
    """Clase para analizar y comparar anotaciones MEDDOCAN"""

    def __init__(self):
        self.tag_errors = defaultdict(lambda: {'fp': [], 'fn': [], 'tp': []})
        self.tag_sentences = defaultdict(int)
        self.results = {}

    def parse_i2b2_annotations(self, xml_file):
        """Parsea las anotaciones de un archivo XML i2b2"""
        annotations = []
        tree = etree.parse(xml_file)
        root = tree.getroot()

        text = root.find("TEXT").text if root.find("TEXT") is not None else ""

        for tag in root.find("TAGS"):
            annotation = {
                'type': tag.tag,
                'start': int(tag.get('start')),
                'end': int(tag.get('end')),
                'text': tag.get('text'),
                'tag_text': text[int(tag.get('start')):int(tag.get('end'))]
            }
            if 'TYPE' in tag.attrib:
                annotation['subtype'] = tag.get('TYPE')
            annotations.append(annotation)

        return {
            'text': text,
            'annotations': annotations
        }

    def get_context(self, text, start, end, window=30):
        """Obtiene el contexto alrededor de una anotación"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        before = text[context_start:start]
        annotated = text[start:end]
        after = text[end:context_end]
        return f"...{before}>>>{annotated}<<<{after}..."

    def compare_document_annotations(self, gold, system, tag_errors, out_f=None):
        """Compara las anotaciones de un documento"""
        gold_ann = gold['annotations']
        sys_ann = system['annotations']
        false_positives = []
        false_negatives = []

        # Buscar True Positives y False Positives
        for s_ann in sys_ann:
            matched = False
            for g_ann in gold_ann:
                if (s_ann['start'] == g_ann['start'] and
                        s_ann['end'] == g_ann['end'] and
                        s_ann.get('subtype', None) == g_ann.get('subtype', None)):
                    matched = True
                    tag_type = g_ann.get('subtype', g_ann['type'])
                    tag_errors[tag_type]['tp'].append({
                        'text': g_ann['text'],
                        'position': (g_ann['start'], g_ann['end'])
                    })
                    break

            if not matched:
                tag_type = s_ann.get('subtype', s_ann['type'])
                tag_errors[tag_type]['fp'].append({
                    'text': system['text'][s_ann['start']:s_ann['end']],
                    'system_text': s_ann['text'],
                    'position': (s_ann['start'], s_ann['end'])
                })
                false_positives.append(s_ann)

        # Buscar False Negatives
        for g_ann in gold_ann:
            matched = False
            for s_ann in sys_ann:
                if (s_ann['start'] == g_ann['start'] and
                        s_ann['end'] == g_ann['end'] and
                        s_ann.get('subtype', None) == g_ann.get('subtype', None)):
                    matched = True
                    break
            if not matched:
                tag_type = g_ann.get('subtype', g_ann['type'])
                tag_errors[tag_type]['fn'].append({
                    'text': gold['text'][g_ann['start']:g_ann['end']],
                    'gold_text': g_ann['text'],
                    'position': (g_ann['start'], g_ann['end'])
                })
                false_negatives.append(g_ann)

        # Escribir errores si se proporciona archivo de salida
        if out_f:
            if false_positives or false_negatives:
                out_f.write("Errors found:\n")
                if false_positives:
                    out_f.write("\nFalse Positives (System tagged but shouldn't):\n")
                    for fp in false_positives:
                        out_f.write(f"- {fp.get('subtype', fp['type'])}: '{fp['text']}' "
                                    f"(pos: {fp['start']}-{fp['end']})\n")
                        out_f.write(f"  Context: '{self.get_context(system['text'], fp['start'], fp['end'])}'\n")

                if false_negatives:
                    out_f.write("\nFalse Negatives (System missed):\n")
                    for fn in false_negatives:
                        out_f.write(f"- {fn.get('subtype', fn['type'])}: '{fn['text']}' "
                                    f"(pos: {fn['start']}-{fn['end']})\n")
                        out_f.write(f"  Context: '{self.get_context(gold['text'], fn['start'], fn['end'])}'\n")
            else:
                out_f.write("No errors found in this document.\n")

        return false_positives, false_negatives

    def leak_score(self, fn, num_sentences):
        """Calcula el leak score"""
        try:
            return round(len(fn) / num_sentences, 4) if num_sentences > 0 else 0.0
        except Exception:
            return "NA"

    def compute_metrics(self, tag_errors, tag_sentences):
        """Computa las métricas por tag"""
        metrics = {}

        for tag, values in tag_errors.items():
            tp = len(values['tp'])
            fp = len(values['fp'])
            fn = len(values['fn'])

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
            leak = self.leak_score(values['fn'], tag_sentences.get(tag, 0))

            metrics[tag] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'leak': leak,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }

        return metrics

    def analyze_annotations(self, gold_dir, system_dir, output_file=None, prompt_name="System"):
        """Analiza las anotaciones y genera reporte"""
        tag_errors = defaultdict(lambda: {'fp': [], 'fn': [], 'tp': []})
        tag_sentences = defaultdict(int)

        output_content = []
        output_content.append(f"MEDDOCAN Error Analysis Report - {prompt_name}")
        output_content.append("=" * 50)

        for filename in os.listdir(gold_dir):
            if filename.endswith(".xml"):
                gold_file = os.path.join(gold_dir, filename)
                system_file = os.path.join(system_dir, filename)

                if not os.path.exists(system_file):
                    continue

                # Parse annotations
                gold_ann = self.parse_i2b2_annotations(gold_file)
                sys_ann = self.parse_i2b2_annotations(system_file)

                output_content.append(f"\nDocument: {filename}")
                output_content.append("-" * 50)

                # Count sentences
                num_sentences = gold_ann['text'].count('.') + gold_ann['text'].count('\n')
                for ann in gold_ann['annotations']:
                    tag_type = ann.get('subtype', ann['type'])
                    tag_sentences[tag_type] += num_sentences

                # Compare annotations
                if output_file:
                    with open(output_file, 'a') as out_f:
                        self.compare_document_annotations(gold_ann, sys_ann, tag_errors, out_f)
                else:
                    self.compare_document_annotations(gold_ann, sys_ann, tag_errors)

        # Compute metrics
        metrics = self.compute_metrics(tag_errors, tag_sentences)

        # Store results
        self.results[prompt_name] = {
            'metrics': metrics,
            'tag_errors': tag_errors,
            'tag_sentences': tag_sentences
        }

        return metrics

    def save_metrics_to_excel(self, metrics, output_path, prompt_name="System"):
        """Guarda las métricas en un archivo Excel"""
        tags, precisions, recalls, f1s, leaks = [], [], [], [], []

        for tag, values in metrics.items():
            tags.append(tag)
            precisions.append(values['precision'])
            recalls.append(values['recall'])
            f1s.append(values['f1'])
            leaks.append(values['leak'])

        metrics_df = pd.DataFrame({
            'Tag': tags,
            'Precision': precisions,
            'Recall': recalls,
            'F1 Score': f1s,
            'Leak Score': leaks
        })

        excel_path = Path(output_path) / f"metrics_{prompt_name.lower().replace(' ', '_')}.xlsx"
        metrics_df.to_excel(excel_path, index=False)

        # Format Excel
        wb = load_workbook(excel_path)
        ws = wb.active

        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill("solid", fgColor="4F81BD")
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill

        alt_fill = PatternFill("solid", fgColor="DCE6F1")
        for row in range(2, ws.max_row + 1):
            if row % 2 == 0:
                for col in range(1, ws.max_column + 1):
                    ws[f"{get_column_letter(col)}{row}"].fill = alt_fill

        for col in ws.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            ws.column_dimensions[column].width = max_length + 2

        wb.save(excel_path)
        return excel_path

    def compare_multiple_prompts(self, comparisons, output_dir="./results"):
        """Compara múltiples prompts y genera visualizaciones"""
        os.makedirs(output_dir, exist_ok=True)

        all_metrics = {}

        # Analizar cada prompt
        for prompt_name, (gold_dir, system_dir) in comparisons.items():
            print(f"Analizando {prompt_name}...")
            metrics = self.analyze_annotations(gold_dir, system_dir, prompt_name=prompt_name)
            all_metrics[prompt_name] = metrics
            self.save_metrics_to_excel(metrics, output_dir, prompt_name)

        # Crear DataFrame consolidado
        consolidated_data = []
        for prompt_name, metrics in all_metrics.items():
            for tag, values in metrics.items():
                consolidated_data.append({
                    'Prompt': prompt_name,
                    'Tag': tag,
                    'Precision': values['precision'],
                    'Recall': values['recall'],
                    'F1': values['f1'],
                    'Leak': values['leak'],
                    'TP': values['tp'],
                    'FP': values['fp'],
                    'FN': values['fn']
                })

        df_consolidated = pd.DataFrame(consolidated_data)
        df_consolidated.to_excel(Path(output_dir) / "consolidated_metrics.xlsx", index=False)

        # Generar visualizaciones
        self.create_comparison_plots(df_consolidated, output_dir)

        return df_consolidated

    def create_comparison_plots(self, df, output_dir):
        """Crea gráficos de comparación"""
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")

        # Gráfico de F1 por tag y prompt
        plt.figure(figsize=(15, 8))
        pivot_f1 = df.pivot(index='Tag', columns='Prompt', values='F1')
        sns.heatmap(pivot_f1, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('F1 Score Comparison by Tag and Prompt')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'f1_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Gráfico de barras agrupadas para F1
        plt.figure(figsize=(15, 8))
        sns.barplot(data=df, x='Tag', y='F1', hue='Prompt')
        plt.title('F1 Score Comparison by Tag')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'f1_barplot.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Gráfico de dispersión Precision vs Recall
        plt.figure(figsize=(12, 8))
        for prompt in df['Prompt'].unique():
            prompt_data = df[df['Prompt'] == prompt]
            plt.scatter(prompt_data['Precision'], prompt_data['Recall'],
                        label=prompt, alpha=0.7, s=100)

        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision vs Recall by Prompt')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'precision_recall_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Resumen estadístico por prompt
        plt.figure(figsize=(12, 6))
        summary_stats = df.groupby('Prompt')[['Precision', 'Recall', 'F1']].mean()
        summary_stats.plot(kind='bar', ax=plt.gca())
        plt.title('Average Metrics by Prompt')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'average_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

        return summary_stats


# Función de conveniencia para uso directo
def analyze_single_prompt(gold_dir, system_dir, output_file=None, prompt_name="System"):
    """Función de conveniencia para analizar un solo prompt"""
    analyzer = MeddocanAnalyzer()
    return analyzer.analyze_annotations(gold_dir, system_dir, output_file, prompt_name)


def compare_prompts(comparisons, output_dir="./results"):
    """Función de conveniencia para comparar múltiples prompts"""
    analyzer = MeddocanAnalyzer()
    return analyzer.compare_multiple_prompts(comparisons, output_dir)