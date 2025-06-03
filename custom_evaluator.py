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
MatchResult = namedtuple('MatchResult', ['tp', 'fp', 'fn', 'phase_info', 'fn_overlap_stats'])
PhaseInfo = namedtuple('PhaseInfo', ['exact_matches', 'inclusion_matches', 'groupings', 'additional_fps'])
FNOverlapStats = namedtuple('FNOverlapStats', ['individual_overlaps', 'max_overlaps', 'distribution'])


class ImprovedEvaluator:
    def __init__(self, margin: float = 0.0, ignore_tags: bool = False, verbose: bool = False):
        self.margin = margin
        self.ignore_tags = ignore_tags
        self.verbose = verbose
        nltk.download('stopwords')
        nltk.download('punkt')
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

    def calculate_overlap_percentage(self, entity1: Entity, entity2: Entity) -> float:
        """Calcula el porcentaje de solapamiento entre dos entidades."""
        # Calcular intersección
        overlap_start = max(entity1.start, entity2.start)
        overlap_end = min(entity1.end, entity2.end)

        if overlap_start >= overlap_end:
            return 0.0

        overlap_length = overlap_end - overlap_start
        entity1_length = entity1.end - entity1.start

        if entity1_length == 0:
            return 0.0

        return (overlap_length / entity1_length) * 100

    def analyze_fn_overlaps(self, fn_entities: List[Entity], sys_entities: List[Entity]) -> FNOverlapStats:
        """Analiza los solapamientos de los False Negatives con las entidades del sistema."""
        individual_overlaps = []
        max_overlaps = []

        for fn_entity in fn_entities:
            fn_overlaps = []

            for sys_entity in sys_entities:
                # Solo considerar entidades de la misma etiqueta si no ignoramos tags
                if not self.ignore_tags and fn_entity.tag != sys_entity.tag:
                    continue

                overlap_pct = self.calculate_overlap_percentage(fn_entity, sys_entity)
                if overlap_pct > 0:
                    fn_overlaps.append({
                        'fn_entity': fn_entity,
                        'sys_entity': sys_entity,
                        'overlap_percentage': overlap_pct
                    })

            individual_overlaps.extend(fn_overlaps)

            # Obtener el máximo solapamiento para este FN
            if fn_overlaps:
                max_overlap = max(fn_overlaps, key=lambda x: x['overlap_percentage'])
                max_overlaps.append({
                    'fn_entity': fn_entity,
                    'max_overlap_percentage': max_overlap['overlap_percentage'],
                    'best_sys_entity': max_overlap['sys_entity']
                })
            else:
                max_overlaps.append({
                    'fn_entity': fn_entity,
                    'max_overlap_percentage': 0.0,
                    'best_sys_entity': None
                })

        # Crear distribución por rangos
        distribution = self.create_overlap_distribution([mo['max_overlap_percentage'] for mo in max_overlaps])

        return FNOverlapStats(
            individual_overlaps=individual_overlaps,
            max_overlaps=max_overlaps,
            distribution=distribution
        )

    def create_overlap_distribution(self, overlap_percentages: List[float]) -> Dict[str, int]:
        """Crea distribución de solapamientos por rangos de 10%."""
        ranges = [
            ("0-10%", 0, 10),
            ("10-20%", 10, 20),
            ("20-30%", 20, 30),
            ("30-40%", 30, 40),
            ("40-50%", 40, 50),
            ("50-60%", 50, 60),
            ("60-70%", 60, 70),
            ("70-80%", 70, 80),
            ("80-90%", 80, 90),
            ("90-100%", 90, 100)
        ]

        distribution = {range_name: 0 for range_name, _, _ in ranges}

        for pct in overlap_percentages:
            for range_name, min_val, max_val in ranges:
                if min_val <= pct < max_val or (pct == 100 and max_val == 100):
                    distribution[range_name] += 1
                    break

        return distribution

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

    def get_non_gold_text(self, sys_text: str, gold_texts: List[str]) -> str:
        """Obtiene el texto en sys_text que no está cubierto por ninguna gold_text."""
        covered_ranges = []
        for gold_text in gold_texts:
            start = 0
            while True:
                pos = sys_text.find(gold_text, start)
                if pos == -1:
                    break
                covered_ranges.append((pos, pos + len(gold_text)))
                start = pos + len(gold_text)

        covered_ranges.sort()
        non_gold_text = []
        last_end = 0

        for start, end in covered_ranges:
            if start > last_end:
                non_gold_text.append(sys_text[last_end:start])
            last_end = max(last_end, end)

        if last_end < len(sys_text):
            non_gold_text.append(sys_text[last_end:])

        return ' '.join(non_gold_text)

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
                              matched_sys: Set[int], matched_gold: Set[int]) -> Tuple[
        Set[int], Set[int], List[Dict], Set[int]]:
        """Fase 2: Inclusión completa con margen y FP por texto adicional significativo."""
        inclusion_matched_sys = set()
        inclusion_matched_gold = set()
        groupings = []
        additional_fps = set()

        for i, sys_entity in enumerate(sys_entities):
            if i in matched_sys:
                continue

            contained_gold = []
            for j, gold_entity in enumerate(gold_entities):
                if (j not in matched_gold and
                        self.is_fully_contained(gold_entity, sys_entity)):
                    if not self.ignore_tags and sys_entity.tag != gold_entity.tag:
                        continue
                    contained_gold.append(j)

            if contained_gold:
                gold_ents = [gold_entities[j] for j in contained_gold]
                ratio = self.calculate_length_ratio(gold_ents, sys_entity)

                gold_texts = [e.text for e in gold_ents]
                additional_text = self.get_non_gold_text(sys_entity.text, gold_texts)
                normalized_additional = self.normalize_text(additional_text)
                has_significant_additional = bool(normalized_additional.strip())

                grouping_info = {
                    'sys_idx': i,
                    'gold_indices': contained_gold,
                    'ratio': ratio,
                    'within_margin': self.is_within_margin(ratio),
                    'sys_entity': sys_entity,
                    'gold_entities': gold_ents,
                    'additional_text': additional_text,
                    'has_significant_additional': has_significant_additional,
                    'normalized_additional': normalized_additional
                }

                if self.is_within_margin(ratio):
                    inclusion_matched_sys.add(i)
                    inclusion_matched_gold.update(contained_gold)

                    if has_significant_additional:
                        additional_fps.add(i)

                groupings.append(grouping_info)

        return inclusion_matched_sys, inclusion_matched_gold, groupings, additional_fps

    def evaluate_document(self, sys_entities: List[Entity], gold_entities: List[Entity]) -> MatchResult:
        """Evalúa un documento con detección de FPs adicionales y análisis de solapamiento de FNs."""
        matched_sys_p1, matched_gold_p1, exact_pairs = self.phase1_exact_matches(sys_entities, gold_entities)
        matched_sys_p2, matched_gold_p2, groupings, additional_fps = self.phase2_full_inclusion(
            sys_entities, gold_entities, matched_sys_p1, matched_gold_p1)

        all_matched_sys = matched_sys_p1.union(matched_sys_p2)
        all_matched_gold = matched_gold_p1.union(matched_gold_p2)

        # TP: gold entities matched
        tp_count = len(all_matched_gold)

        # FP: system entities not matched + additional text in matched entities
        fp_count = (len(sys_entities) - len(all_matched_sys)) + len(additional_fps)

        # FN: gold entities not matched
        fn_count = len(gold_entities) - len(all_matched_gold)

        # Obtener FN entities para análisis de solapamiento
        fn_entities = [ent for i, ent in enumerate(gold_entities) if i not in all_matched_gold]

        # Analizar solapamientos de FNs
        fn_overlap_stats = self.analyze_fn_overlaps(fn_entities, sys_entities)

        phase_info = PhaseInfo(
            exact_matches=exact_pairs,
            inclusion_matches=list(zip(matched_sys_p2,
                                       [g['gold_indices'] for g in groupings if g['within_margin']])),
            groupings=groupings,
            additional_fps=additional_fps
        )

        return MatchResult(tp=tp_count, fp=fp_count, fn=fn_count, phase_info=phase_info,
                           fn_overlap_stats=fn_overlap_stats)

    def extract_entities_from_annotation(self, annotation) -> List[Entity]:
        """Extrae entidades de una anotación."""
        entities = []

        if hasattr(annotation, 'phi') and annotation.phi:
            for phi in annotation.phi:
                if hasattr(phi, '__iter__') and len(phi) >= 3:
                    tag, start, end = phi[0], phi[1], phi[2]
                    text = annotation.text[start:end] if annotation.text else ""
                else:
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
        report.append(
            f"  False Positives: {result.fp} (incluye {len(getattr(result.phase_info, 'additional_fps', []))} por texto adicional)")
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
                reason = ""
                if not group['within_margin']:
                    reason = " (fuera de margen)"
                elif group['has_significant_additional']:
                    reason = " (pero con FP adicional)"

                report.append(f"  {status}{reason} (ratio: {group['ratio']:.3f})")
                report.append(
                    f"    Sistema: [{group['sys_entity'].start}-{group['sys_entity'].end}] '{group['sys_entity'].text}'")
                for gold_ent in group['gold_entities']:
                    report.append(f"    Gold:    [{gold_ent.start}-{gold_ent.end}] '{gold_ent.text}'")

                if group['additional_text']:
                    report.append(f"    Texto adicional: '{group['additional_text']}'")
                    if group['has_significant_additional']:
                        report.append(f"    -> Contiene texto significativo adicional (FP)")

        # Falsos positivos adicionales
        if hasattr(result.phase_info, 'additional_fps') and result.phase_info.additional_fps:
            report.append(f"\nFALSOS POSITIVOS ADICIONALES EN AGRUPAMIENTOS ({len(result.phase_info.additional_fps)}):")
            for i in result.phase_info.additional_fps:
                sys_ent = sys_entities[i]
                for group in result.phase_info.groupings:
                    if group['sys_idx'] == i:
                        report.append(f"  ✗ [{sys_ent.start}-{sys_ent.end}] '{sys_ent.text}'")
                        report.append(f"     Texto adicional significativo: '{group['normalized_additional']}'")
                        break

        # Falsos positivos no emparejados
        matched_sys_indices = set()
        for sys_idx, _ in result.phase_info.exact_matches:
            matched_sys_indices.add(sys_idx)
        for group in result.phase_info.groupings:
            if group['within_margin']:
                matched_sys_indices.add(group['sys_idx'])

        fp_entities = [ent for i, ent in enumerate(sys_entities) if
                       i not in matched_sys_indices and i not in result.phase_info.additional_fps]
        if fp_entities:
            report.append(f"\nFALSOS POSITIVOS NO EMPAREJADOS ({len(fp_entities)}):")
            for ent in fp_entities:
                report.append(f"  ✗ [{ent.start}-{ent.end}] '{ent.text}' ({ent.tag})")

        # Falsos negativos con análisis de solapamiento
        if result.fn_overlap_stats.max_overlaps:
            report.append(
                f"\nFALSOS NEGATIVOS CON ANÁLISIS DE SOLAPAMIENTO ({len(result.fn_overlap_stats.max_overlaps)}):")
            for overlap_info in result.fn_overlap_stats.max_overlaps:
                fn_ent = overlap_info['fn_entity']
                max_pct = overlap_info['max_overlap_percentage']
                best_sys = overlap_info['best_sys_entity']

                if max_pct > 0:
                    report.append(
                        f"  ✗ [{fn_ent.start}-{fn_ent.end}] '{fn_ent.text}' ({fn_ent.tag}) - Máximo solapamiento: {max_pct:.1f}%")
                    report.append(
                        f"     Mejor match sistema: [{best_sys.start}-{best_sys.end}] '{best_sys.text}' ({best_sys.tag})")
                else:
                    report.append(
                        f"  ✗ [{fn_ent.start}-{fn_ent.end}] '{fn_ent.text}' ({fn_ent.tag}) - Sin solapamiento")

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

        # Acumular estadísticas de solapamiento de FN
        all_fn_overlap_distributions = []

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

            # Acumular distribuciones de solapamiento
            all_fn_overlap_distributions.append(result.fn_overlap_stats.distribution)

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

        # Combinar distribuciones de solapamiento de todos los documentos
        combined_distribution = {}
        for distribution in all_fn_overlap_distributions:
            for range_name, count in distribution.items():
                combined_distribution[range_name] = combined_distribution.get(range_name, 0) + count

        all_results[system_name] = {
            'metrics': total_metrics,
            'doc_results': doc_results,
            'error_reports': error_reports,
            'fn_overlap_distribution': combined_distribution
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

            # Archivo de métricas con estadísticas de solapamiento
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

                # Estadísticas de solapamiento de False Negatives
                f.write(f"ESTADÍSTICAS DE SOLAPAMIENTO DE FALSE NEGATIVES:\n")
                f.write(f"{'=' * 50}\n")
                total_fn_analyzed = sum(combined_distribution.values())
                if total_fn_analyzed > 0:
                    f.write(f"Total de FN analizados: {total_fn_analyzed}\n\n")
                    f.write("Distribución por rangos de solapamiento:\n")
                    f.write("-" * 40 + "\n")
                    for range_name in ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                                       "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]:
                        count = combined_distribution.get(range_name, 0)
                        percentage = (count / total_fn_analyzed) * 100 if total_fn_analyzed > 0 else 0
                        f.write(f"{range_name:>8}: {count:>4} ({percentage:>5.1f}%)\n")
                else:
                    f.write("No hay False Negatives para analizar.\n")

                if verbose:
                    f.write(f"\nMétricas por Documento:\n")
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