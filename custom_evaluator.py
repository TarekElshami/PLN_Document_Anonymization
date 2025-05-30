import os
import argparse
import re
from collections import namedtuple
from typing import List, Dict, Tuple, Set, Optional, Any
from classes import i2b2Annotation, BratAnnotation
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Definir estructuras de datos
Entity = namedtuple('Entity', ['text', 'start', 'end', 'tag', 'doc_id'])
MatchResult = namedtuple('MatchResult', ['tp', 'fp', 'fn', 'phase_info'])
PhaseInfo = namedtuple('PhaseInfo', ['exact_matches', 'inclusion_matches', 'partial_matches', 'groupings'])

class ImprovedEvaluator:
    def __init__(self, margin: float = 0.0, ignore_tags: bool = False, verbose: bool = False):
        self.margin = margin
        self.ignore_tags = ignore_tags
        self.verbose = verbose
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('punkt_tab')
        self.stopwords = set(stopwords.words('spanish'))
        self.punctuation_pattern = re.compile(r'[^\w\s]', re.UNICODE)


    def normalize_text(self, text):
        """Normaliza texto usando NLTK para tokenización y stopwords"""
        if not text:
            return ""

        text = self.punctuation_pattern.sub(' ', text)

        # Tokenización con NLTK
        tokens = word_tokenize(text, language='spanish')

        # Filtrar stopwords y términos vacíos
        tokens = [token for token in tokens
                  if token not in self.stopwords
                  and token.strip()
                  and len(token) > 1]

        return ' '.join(tokens)


    def entities_match(self, sys_entity: Entity, gold_entity: Entity) -> bool:
        """Verifica si dos entidades coinciden según el modo de evaluación."""
        # Verificar posición
        if sys_entity.start != gold_entity.start or sys_entity.end != gold_entity.end:
            return False

        # Verificar texto
        if sys_entity.text != gold_entity.text:
            return False

        # Verificar etiqueta si no se ignoran
        if not self.ignore_tags and sys_entity.tag != gold_entity.tag:
            return False

        return True


    def is_fully_contained(self, inner_entity: Entity, outer_entity: Entity) -> bool:
        """Verifica si inner_entity está completamente contenida en outer_entity."""
        return (outer_entity.start <= inner_entity.start and
                inner_entity.end <= outer_entity.end)


    def calculate_length_ratio(self, gold_entities: List[Entity], sys_entity: Entity) -> float:
        """Calcula la proporción de longitudes entre entidades gold y sistema."""
        gold_total_length = sum(len(self.normalize_text(e.text)) for e in gold_entities)
        sys_length = len(self.normalize_text(sys_entity.text))

        if sys_length == 0:
            return float('inf') if gold_total_length > 0 else 1.0

        return gold_total_length / sys_length

    def is_within_margin(self, ratio: float) -> bool:
        """Verifica si la proporción está dentro del margen aceptable."""
        return abs(ratio - 1.0) <= self.margin

    def phase1_exact_matches(self, sys_entities: List[Entity], gold_entities: List[Entity]) -> Tuple[
        Set[int], Set[int], List[Tuple[int, int]]]:
        """Fase 1: Encuentra coincidencias exactas."""
        matched_sys = set()
        matched_gold = set()
        exact_pairs = []

        for i, sys_entity in enumerate(sys_entities):
            for j, gold_entity in enumerate(gold_entities):
                if (i not in matched_sys and j not in matched_gold and
                        self.entities_match(sys_entity, gold_entity)):
                    matched_sys.add(i)
                    matched_gold.add(j)
                    exact_pairs.append((i, j))
                    break

        return matched_sys, matched_gold, exact_pairs

    def phase2_full_inclusion(self, sys_entities: List[Entity], gold_entities: List[Entity],
                              matched_sys: Set[int], matched_gold: Set[int]) -> Tuple[Set[int], Set[int], List[Dict]]:
        """Fase 2: Inclusión completa con margen."""
        inclusion_matched_sys = set()
        inclusion_matched_gold = set()
        groupings = []

        # Buscar entidades gold contenidas en entidades sistema
        for i, sys_entity in enumerate(sys_entities):
            if i in matched_sys:  # Ya emparejada en fase 1
                continue

            contained_gold = []
            for j, gold_entity in enumerate(gold_entities):
                if (j not in matched_gold and
                        self.is_fully_contained(gold_entity, sys_entity)):
                    # Verificar etiquetas si no se ignoran
                    if not self.ignore_tags and sys_entity.tag != gold_entity.tag:
                        continue
                    contained_gold.append(j)

            if contained_gold:
                # Calcular proporción de longitudes
                gold_ents = [gold_entities[j] for j in contained_gold]
                ratio = self.calculate_length_ratio(gold_ents, sys_entity)

                grouping_info = {
                    'sys_idx': i,
                    'gold_indices': contained_gold,
                    'ratio': ratio,
                    'within_margin': self.is_within_margin(ratio),
                    'sys_entity': sys_entity,
                    'gold_entities': gold_ents
                }

                if self.is_within_margin(ratio):
                    inclusion_matched_sys.add(i)
                    inclusion_matched_gold.update(contained_gold)

                groupings.append(grouping_info)

        return inclusion_matched_sys, inclusion_matched_gold, groupings

    def evaluate_document(self, sys_entities: List[Entity], gold_entities: List[Entity]) -> MatchResult:
        """Evalúa un documento usando las tres fases."""

        # Fase 1: Coincidencias exactas
        matched_sys_p1, matched_gold_p1, exact_pairs = self.phase1_exact_matches(sys_entities, gold_entities)

        # Fase 2: Inclusión completa
        matched_sys_p2, matched_gold_p2, groupings = self.phase2_full_inclusion(
            sys_entities, gold_entities, matched_sys_p1, matched_gold_p1)

        # Combinar matches de ambas fases
        all_matched_sys = matched_sys_p1.union(matched_sys_p2)
        all_matched_gold = matched_gold_p1.union(matched_gold_p2)

        # Calcular métricas
        tp_count = len(all_matched_gold)
        fp_count = len(sys_entities) - len(all_matched_sys)
        fn_count = len(gold_entities) - len(all_matched_gold)

        # Crear información de fases
        phase_info = PhaseInfo(
            exact_matches=exact_pairs,
            inclusion_matches=list(zip(matched_sys_p2,
                                       [g['gold_indices'] for g in groupings if g['within_margin']])),
            partial_matches=[],
            groupings=groupings
        )

        return MatchResult(tp=tp_count, fp=fp_count, fn=fn_count, phase_info=phase_info)

    def extract_entities_from_annotation(self, annotation) -> List[Entity]:
        """Extrae entidades de una anotación."""
        entities = []

        if hasattr(annotation, 'phi') and annotation.phi:
            for phi in annotation.phi:
                if hasattr(phi, '__iter__') and len(phi) >= 3:
                    # Formato BRAT: (tag, start, end)
                    tag, start, end = phi[0], phi[1], phi[2]
                    text = annotation.text[start:end] if annotation.text else ""
                else:
                    # Formato i2b2: objeto PHI con métodos
                    tag = getattr(phi, 'TYPE', 'UNKNOWN')
                    start = phi.get_start() if hasattr(phi, 'get_start') else 0
                    end = phi.get_end() if hasattr(phi, 'get_end') else 0
                    text = annotation.text[start:end] if annotation.text else ""

                entities.append(Entity(
                    text=text,
                    start=start,
                    end=end,
                    tag=tag,
                    doc_id=annotation.doc_id
                ))

        return entities

    def calculate_metrics(self, tp: int, fp: int, fn: int) -> Dict[str, float]:
        """Calcula precision, recall y F1."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def generate_error_report(self, doc_id: str, sys_entities: List[Entity],
                              gold_entities: List[Entity], result: MatchResult) -> str:
        """Genera reporte detallado de errores para un documento."""
        report = [f"\n{'=' * 60}"]
        report.append(f"DOCUMENTO: {doc_id}")
        report.append(f"{'=' * 60}")

        # Resumen
        report.append(f"\nRESUMEN:")
        report.append(f"  Entidades Sistema: {len(sys_entities)}")
        report.append(f"  Entidades Gold: {len(gold_entities)}")
        report.append(f"  True Positives: {result.tp}")
        report.append(f"  False Positives: {result.fp}")
        report.append(f"  False Negatives: {result.fn}")

        metrics = self.calculate_metrics(result.tp, result.fp, result.fn)
        report.append(f"  Precision: {metrics['precision']:.4f}")
        report.append(f"  Recall: {metrics['recall']:.4f}")
        report.append(f"  F1: {metrics['f1']:.4f}")

        # Fase 1: Coincidencias exactas
        if result.phase_info.exact_matches:
            report.append(f"\nFASE 1 - COINCIDENCIAS EXACTAS ({len(result.phase_info.exact_matches)}):")
            for sys_idx, gold_idx in result.phase_info.exact_matches:
                sys_ent = sys_entities[sys_idx]
                gold_ent = gold_entities[gold_idx]
                report.append(f"  ✓ [{sys_ent.start}-{sys_ent.end}] '{sys_ent.text}' ({sys_ent.tag})")

        # Fase 2: Agrupamientos
        if result.phase_info.groupings:
            report.append(f"\nFASE 2 - AGRUPAMIENTOS:")
            for group in result.phase_info.groupings:
                status = "✓ ACEPTADO" if group['within_margin'] else "✗ RECHAZADO"
                report.append(f"  {status} (ratio: {group['ratio']:.3f})")
                report.append(
                    f"    Sistema: [{group['sys_entity'].start}-{group['sys_entity'].end}] '{group['sys_entity'].text}'")
                for gold_ent in group['gold_entities']:
                    report.append(f"    Gold:    [{gold_ent.start}-{gold_ent.end}] '{gold_ent.text}'")

        # Falsos positivos
        matched_sys_indices = set()
        for sys_idx, _ in result.phase_info.exact_matches:
            matched_sys_indices.add(sys_idx)
        for group in result.phase_info.groupings:
            if group['within_margin']:
                matched_sys_indices.add(group['sys_idx'])

        fp_entities = [ent for i, ent in enumerate(sys_entities) if i not in matched_sys_indices]
        if fp_entities:
            report.append(f"\nFALSOS POSITIVOS ({len(fp_entities)}):")
            for ent in fp_entities:
                report.append(f"  ✗ [{ent.start}-{ent.end}] '{ent.text}' ({ent.tag})")

        # Falsos negativos
        matched_gold_indices = set()
        for _, gold_idx in result.phase_info.exact_matches:
            matched_gold_indices.add(gold_idx)
        for group in result.phase_info.groupings:
            if group['within_margin']:
                matched_gold_indices.update(group['gold_indices'])

        fn_entities = [ent for i, ent in enumerate(gold_entities) if i not in matched_gold_indices]
        if fn_entities:
            report.append(f"\nFALSOS NEGATIVOS ({len(fn_entities)}):")
            for ent in fn_entities:
                report.append(f"  ✗ [{ent.start}-{ent.end}] '{ent.text}' ({ent.tag})")

        return '\n'.join(report)


def load_annotations(path: str, annotation_class) -> Dict[str, Any]:
    """Carga anotaciones desde archivo o directorio."""
    annotations = {}

    if os.path.isfile(path):
        # Archivo individual
        ann = annotation_class(path)
        annotations[ann.doc_id] = ann
    elif os.path.isdir(path):
        # Directorio
        for filename in os.listdir(path):
            if filename.endswith('.xml') or filename.endswith('.ann'):
                filepath = os.path.join(path, filename)
                try:
                    ann = annotation_class(filepath)
                    annotations[ann.doc_id] = ann
                except Exception as e:
                    print(f"Error cargando {filepath}: {e}")
    else:
        raise ValueError(f"Ruta no válida: {path}")

    return annotations


def evaluate_with_margin(gs_path: str, sys_paths: List[str], annotation_class,
                         margin: float = 0.0, ignore_tags: bool = False,
                         output_dir: Optional[str] = None, verbose: bool = False):
    """Función principal de evaluación con márgenes."""

    evaluator = ImprovedEvaluator(margin=margin, ignore_tags=ignore_tags, verbose=verbose)

    # Cargar gold standard
    print(f"Cargando gold standard desde: {gs_path}")
    gold_annotations = load_annotations(gs_path, annotation_class)
    print(f"Cargadas {len(gold_annotations)} anotaciones gold")

    all_results = {}

    # Procesar cada sistema
    for sys_path in sys_paths:
        system_name = os.path.basename(sys_path.rstrip('/'))
        print(f"\nEvaluando sistema: {system_name}")

        # Cargar anotaciones del sistema
        sys_annotations = load_annotations(sys_path, annotation_class)
        print(f"Cargadas {len(sys_annotations)} anotaciones del sistema")

        # Encontrar documentos comunes
        common_docs = set(gold_annotations.keys()) & set(sys_annotations.keys())
        if not common_docs:
            print(f"¡Advertencia! No hay documentos comunes entre gold y sistema {system_name}")
            continue

        print(f"Evaluando {len(common_docs)} documentos comunes")

        # Evaluar cada documento
        doc_results = {}
        total_tp, total_fp, total_fn = 0, 0, 0
        error_reports = []

        for doc_id in sorted(common_docs):
            gold_ann = gold_annotations[doc_id]
            sys_ann = sys_annotations[doc_id]

            # Extraer entidades
            gold_entities = evaluator.extract_entities_from_annotation(gold_ann)
            sys_entities = evaluator.extract_entities_from_annotation(sys_ann)

            # Evaluar documento
            result = evaluator.evaluate_document(sys_entities, gold_entities)
            doc_results[doc_id] = result

            # Acumular totales
            total_tp += result.tp
            total_fp += result.fp
            total_fn += result.fn

            # Generar reporte de errores
            if output_dir:
                error_report = evaluator.generate_error_report(doc_id, sys_entities, gold_entities, result)
                error_reports.append(error_report)

            # Mostrar resultados por documento si es verbose
            if verbose:
                metrics = evaluator.calculate_metrics(result.tp, result.fp, result.fn)
                print(f"  {doc_id}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

        # Calcular métricas totales
        total_metrics = evaluator.calculate_metrics(total_tp, total_fp, total_fn)
        all_results[system_name] = {
            'metrics': total_metrics,
            'doc_results': doc_results,
            'error_reports': error_reports
        }

        # Mostrar resumen del sistema
        print(f"\nRESULTADOS SISTEMA: {system_name}")
        print(f"{'=' * 50}")
        print(f"Precision: {total_metrics['precision']:.4f}")
        print(f"Recall:    {total_metrics['recall']:.4f}")
        print(f"F1-score:  {total_metrics['f1']:.4f}")
        print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")

        # Guardar reportes si se especificó directorio
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Archivo de errores
            error_file = os.path.join(output_dir, f"errores_{system_name}.txt")
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"REPORTE DE ERRORES - SISTEMA: {system_name}\n")
                f.write(f"Margen: {margin:.2%}, Modo: {'RELAJADO' if ignore_tags else 'ESTRICTO'}\n")
                f.write(''.join(error_reports))

            # Archivo de métricas
            metrics_file = os.path.join(output_dir, f"metrics_{system_name}.txt")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                f.write(f"MÉTRICAS - SISTEMA: {system_name}\n")
                f.write(f"{'=' * 40}\n")
                f.write(f"Configuración:\n")
                f.write(f"  Margen: {margin:.2%}\n")
                f.write(
                    f"  Modo: {'RELAJADO (ignora etiquetas)' if ignore_tags else 'ESTRICTO (considera etiquetas)'}\n")
                f.write(f"  Documentos evaluados: {len(common_docs)}\n\n")
                f.write(f"Métricas Agregadas:\n")
                f.write(f"  Precision: {total_metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {total_metrics['recall']:.4f}\n")
                f.write(f"  F1-score:  {total_metrics['f1']:.4f}\n")
                f.write(f"  TP: {total_tp}\n")
                f.write(f"  FP: {total_fp}\n")
                f.write(f"  FN: {total_fn}\n\n")

                if verbose:
                    f.write("Métricas por Documento:\n")
                    f.write("-" * 60 + "\n")
                    for doc_id in sorted(doc_results.keys()):
                        result = doc_results[doc_id]
                        doc_metrics = evaluator.calculate_metrics(result.tp, result.fp, result.fn)
                        f.write(
                            f"{doc_id:30} P={doc_metrics['precision']:.3f} R={doc_metrics['recall']:.3f} F1={doc_metrics['f1']:.3f}\n")

            print(f"Reportes guardados en: {output_dir}")

    return all_results


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Improved MEDDOCAN Margin Evaluator")

    parser.add_argument("format",
                        choices=["i2b2", "brat"],
                        help="Annotation format")
    parser.add_argument("gs_dir",
                        help="Directory or file with gold standard")
    parser.add_argument("sys_dir",
                        nargs="+",
                        help="Directories or files with system outputs")
    parser.add_argument("--margin", "-m",
                        type=float,
                        default=0.0,
                        help="Error margin as percentage (e.g., 0.1 for 10%%)")
    parser.add_argument("--ignore-tags",
                        action="store_true",
                        help="Use relaxed evaluation (ignore entity types, only consider spans)")
    parser.add_argument("--output-errors",
                        help="Directory to write detailed error and metrics reports")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Show detailed per-document results")

    args = parser.parse_args()

    # Validar margen
    if args.margin < 0 or args.margin > 1:
        raise ValueError("Margin must be between 0 and 1 (0% to 100%)")

    # Mostrar configuración
    print(f"Running IMPROVED evaluation with margin: {args.margin:.2%}")
    print(f"Evaluation mode: {'RELAXED (ignore tags)' if args.ignore_tags else 'STRICT (consider tags)'}")
    if args.output_errors:
        print(f"Reports will be saved to: {args.output_errors}")
    print()

    evaluate_with_margin(
        gs_path=args.gs_dir,
        sys_paths=args.sys_dir,
        annotation_class=i2b2Annotation if args.format == "i2b2" else BratAnnotation,
        margin=args.margin,
        ignore_tags=args.ignore_tags,
        output_dir=args.output_errors,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()