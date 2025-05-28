###############################################################################
#
#   Evaluador MEDDOCAN con Margen de Error
#
#   Este evaluador permite especificar un margen de error porcentual para
#   evaluar las predicciones del sistema con mayor flexibilidad.
#
#   Uso:
#   python margin_evaluator.py [i2b2|brat] GOLD SYSTEM --margin 0.1 [--ignore-tags]
#
###############################################################################

import os
import argparse
from classes import i2b2Annotation, BratAnnotation
from collections import defaultdict
import math


class MarginEvaluator:
    """Evaluador base con margen de error porcentual."""

    def __init__(self, sys_ann, gs_ann, margin=0.0, check_tags=True, output_errors=None):
        self.tp = []
        self.fp = []
        self.fn = []
        self.doc_ids = []
        self.margin = margin
        self.check_tags = check_tags
        self.output_errors = output_errors
        self.errors = []

        self.sys_id = sys_ann[list(sys_ann.keys())[0]].sys_id
        self.label = f"Margin Evaluator [margin={margin:.2%}, tags={'ON' if check_tags else 'OFF'}]"

        self._evaluate(sys_ann, gs_ann)

        if self.output_errors:
            self._write_errors()

    def _get_entity_length(self, start, end):
        """Calcula la longitud de una entidad."""
        return end - start

    def _calculate_margin(self, length, margin_percent):
        """Calcula el margen absoluto basado en el porcentaje y la longitud."""
        return max(1, math.ceil(length * margin_percent))

    def _spans_overlap_with_margin(self, span1, span2, margin_percent):
        """
        Comprueba si dos spans se solapan considerando un margen de error.
        El margen se calcula como porcentaje de la longitud del span de referencia (gold).
        """
        gold_start, gold_end = span1
        sys_start, sys_end = span2

        gold_length = self._get_entity_length(gold_start, gold_end)
        margin = self._calculate_margin(gold_length, margin_percent)

        # Expandir el span gold con el margen
        expanded_gold_start = gold_start - margin
        expanded_gold_end = gold_end + margin

        # Comprobar si el span del sistema está contenido en el gold expandido
        return (sys_start >= expanded_gold_start and
                sys_end <= expanded_gold_end)

    def _tags_match(self, tag1, tag2):
        """Comprueba si dos tags coinciden."""
        if not self.check_tags:
            return True

        # Para i2b2, comparar el TYPE
        if hasattr(tag1, 'TYPE') and hasattr(tag2, 'TYPE'):
            return tag1.TYPE == tag2.TYPE

        # Para BRAT, comparar el primer elemento de la tupla
        if isinstance(tag1, tuple) and isinstance(tag2, tuple):
            return tag1[0] == tag2[0]

        return str(tag1) == str(tag2)

    def _get_span_info(self, tag):
        """Extrae información del span según el formato."""
        if hasattr(tag, 'get_start') and hasattr(tag, 'get_end'):
            # i2b2 format
            return (tag.get_start(), tag.get_end(), getattr(tag, 'TYPE', 'UNKNOWN'))
        elif isinstance(tag, tuple) and len(tag) >= 3:
            # BRAT format con tag
            return (tag[1], tag[2], tag[0])
        elif isinstance(tag, tuple) and len(tag) == 2:
            # Solo spans
            return (tag[0], tag[1], 'SPAN')
        else:
            return (0, 0, 'UNKNOWN')

    def _find_matching_entity(self, target_entity, entity_list):
        """Encuentra una entidad que coincida con la target considerando el margen."""
        target_start, target_end, target_type = self._get_span_info(target_entity)

        for entity in entity_list:
            entity_start, entity_end, entity_type = self._get_span_info(entity)

            # Comprobar solapamiento con margen
            if self._spans_overlap_with_margin((target_start, target_end),
                                               (entity_start, entity_end),
                                               self.margin):
                # Si no necesitamos comprobar tags, ya es válido
                if not self.check_tags:
                    return entity

                # Si necesitamos comprobar tags, verificar que coincidan
                if self._tags_match(target_entity, entity):
                    return entity

        return None

    def _evaluate(self, sys_ann, gs_ann):
        """Realiza la evaluación principal."""

        for doc_id in sorted(list(set(sys_ann.keys()) & set(gs_ann.keys()))):
            gold_entities = self._get_entities(gs_ann[doc_id])
            sys_entities = self._get_entities(sys_ann[doc_id])

            tp_doc = []
            fp_doc = []
            fn_doc = []

            # Crear copias para poder modificarlas
            remaining_sys = list(sys_entities)
            remaining_gold = list(gold_entities)

            # Encontrar True Positives
            for gold_entity in gold_entities:
                matching_sys = self._find_matching_entity(gold_entity, remaining_sys)
                if matching_sys is not None:
                    tp_doc.append(gold_entity)
                    remaining_sys.remove(matching_sys)
                    remaining_gold.remove(gold_entity)

            # Los restantes son FP y FN
            fp_doc = remaining_sys
            fn_doc = remaining_gold

            # Registrar errores
            self._record_errors(doc_id, fp_doc, fn_doc, gold_entities, sys_entities)

            self.tp.append(tp_doc)
            self.fp.append(fp_doc)
            self.fn.append(fn_doc)
            self.doc_ids.append(doc_id)

    def _get_entities(self, annotation):
        """Extrae las entidades de la anotación."""
        if hasattr(annotation, 'get_phi'):
            return annotation.get_phi()
        else:
            return []

    def _record_errors(self, doc_id, fp, fn, gold_entities, sys_entities):
        """Registra los errores para escribirlos después."""
        for entity in fp:
            start, end, entity_type = self._get_span_info(entity)
            self.errors.append({
                'doc_id': doc_id,
                'error_type': 'FALSE_POSITIVE',
                'entity_type': entity_type,
                'start': start,
                'end': end,
                'text': f"System predicted but not in gold"
            })

        for entity in fn:
            start, end, entity_type = self._get_span_info(entity)
            self.errors.append({
                'doc_id': doc_id,
                'error_type': 'FALSE_NEGATIVE',
                'entity_type': entity_type,
                'start': start,
                'end': end,
                'text': f"In gold but not predicted by system"
            })

    def _write_errors(self):
        """Escribe los errores a un archivo de texto."""
        with open(self.output_errors, 'w', encoding='utf-8') as f:
            f.write(f"ERROR REPORT\n")
            f.write(f"={'=' * 50}\n")
            f.write(f"Evaluator: {self.label}\n")
            f.write(f"System ID: {self.sys_id}\n")
            f.write(f"Total Errors: {len(self.errors)}\n\n")

            current_doc = None
            for error in self.errors:
                if error['doc_id'] != current_doc:
                    current_doc = error['doc_id']
                    f.write(f"\nDocument: {current_doc}\n")
                    f.write(f"{'-' * 30}\n")

                f.write(f"{error['error_type']}: {error['entity_type']} ")
                f.write(f"[{error['start']}:{error['end']}] - {error['text']}\n")

    @staticmethod
    def precision(tp, fp):
        try:
            return len(tp) / float(len(fp) + len(tp))
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def recall(tp, fn):
        try:
            return len(tp) / float(len(fn) + len(tp))
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def f1_score(precision, recall):
        try:
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.0

    def micro_precision(self):
        try:
            return sum([len(t) for t in self.tp]) / \
                float(sum([len(t) for t in self.tp]) +
                      sum([len(t) for t in self.fp]))
        except ZeroDivisionError:
            return 0.0

    def micro_recall(self):
        try:
            return sum([len(t) for t in self.tp]) / \
                float(sum([len(t) for t in self.tp]) +
                      sum([len(t) for t in self.fn]))
        except ZeroDivisionError:
            return 0.0

    def print_report(self, verbose=False):
        """Imprime el reporte de evaluación."""

        if verbose:
            self._print_docs()

        self._print_summary()

    def _print_docs(self):
        """Imprime estadísticas por documento."""
        print(f"\nDocument-level Results:")
        print(f"{'-' * 70}")
        print(f"{'Document ID':<35}{'Measure':<15}{'Score':<20}")
        print(f"{'-' * 70}")

        for i, doc_id in enumerate(self.doc_ids):
            mp = self.precision(self.tp[i], self.fp[i])
            mr = self.recall(self.tp[i], self.fn[i])
            f1 = self.f1_score(mp, mr)

            print(f"{doc_id:<35}{'Precision':<15}{mp:.4f}")
            print(f"{'':35}{'Recall':<15}{mr:.4f}")
            print(f"{'':35}{'F1':<15}{f1:.4f}")
            print(f"{'-' * 70}")

    def _print_summary(self):
        """Imprime el resumen general."""
        mp = self.micro_precision()
        mr = self.micro_recall()
        f1 = self.f1_score(mp, mr)

        print(f"\nReport ({self.sys_id}):")
        print(f"{'-' * 70}")
        print(f"{self.label}")
        print(f"{'-' * 70}")
        print(f"{'Total ({} docs)'.format(len(self.doc_ids)):<35}{'Precision':<15}{mp:.4f}")
        print(f"{'':35}{'Recall':<15}{mr:.4f}")
        print(f"{'':35}{'F1':<15}{f1:.4f}")
        print(f"{'-' * 70}")
        print()


class MarginEvaluatorWithTags(MarginEvaluator):
    """Evaluador que considera tanto spans como tags."""

    def __init__(self, sys_ann, gs_ann, margin=0.0, output_errors=None):
        super().__init__(sys_ann, gs_ann, margin, check_tags=True, output_errors=output_errors)
        self.label = f"Margin Evaluator WITH TAGS [margin={margin:.2%}]"


class MarginEvaluatorIgnoreTags(MarginEvaluator):
    """Evaluador que ignora los tags, solo considera spans."""

    def __init__(self, sys_ann, gs_ann, margin=0.0, output_errors=None):
        super().__init__(sys_ann, gs_ann, margin, check_tags=False, output_errors=output_errors)
        self.label = f"Margin Evaluator IGNORE TAGS [margin={margin:.2%}]"


def get_document_dict_by_system_id(system_dirs, annotation_format):
    """Carga documentos del sistema."""
    documents = defaultdict(lambda: defaultdict(int))

    for d in system_dirs:
        for fn in os.listdir(d):
            if fn.endswith(".ann") or fn.endswith(".xml"):
                sa = annotation_format(os.path.join(d, fn))
                documents[sa.sys_id][sa.id] = sa

    return documents


def evaluate_with_margin(gs, system, annotation_format, margin=0.0,
                         ignore_tags=False, output_errors=None, verbose=False):
    """Función principal de evaluación con margen."""

    gold_ann = {}
    evaluations = []

    # Manejar archivos individuales
    if os.path.isfile(system[0]) and os.path.isfile(gs):
        if (system[0].endswith(".ann") and gs.endswith(".ann")) or \
                (system[0].endswith(".xml") and gs.endswith(".xml")):
            gs_ann = annotation_format(gs)
            sys_ann = annotation_format(system[0])

            if ignore_tags:
                evaluator = MarginEvaluatorIgnoreTags(
                    {sys_ann.id: sys_ann},
                    {gs_ann.id: gs_ann},
                    margin=margin,
                    output_errors=output_errors
                )
            else:
                evaluator = MarginEvaluatorWithTags(
                    {sys_ann.id: sys_ann},
                    {gs_ann.id: gs_ann},
                    margin=margin,
                    output_errors=output_errors
                )

            evaluator.print_report(verbose=verbose)
            evaluations.append(evaluator)

    # Manejar directorios
    elif all([os.path.isdir(sys) for sys in system]) and os.path.isdir(gs):
        # Cargar gold standard
        for filename in os.listdir(gs):
            if filename.endswith(".ann") or filename.endswith(".xml"):
                annotations = annotation_format(os.path.join(gs, filename))
                gold_ann[annotations.id] = annotations

        # Evaluar cada sistema
        for system_id, system_ann in sorted(get_document_dict_by_system_id(system, annotation_format).items()):
            if ignore_tags:
                evaluator = MarginEvaluatorIgnoreTags(
                    system_ann,
                    gold_ann,
                    margin=margin,
                    output_errors=output_errors
                )
            else:
                evaluator = MarginEvaluatorWithTags(
                    system_ann,
                    gold_ann,
                    margin=margin,
                    output_errors=output_errors
                )

            evaluator.print_report(verbose=verbose)
            evaluations.append(evaluator)

    else:
        raise Exception("Must pass file file or [directory/]+ directory/ on command line!")

    return evaluations[0] if len(evaluations) == 1 else evaluations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEDDOCAN Margin Evaluator")

    parser.add_argument("format",
                        choices=["i2b2", "brat"],
                        help="Annotation format")

    parser.add_argument("gs_dir",
                        help="Directory or file with gold standard")

    parser.add_argument("sys_dir",
                        help="Directories or files with system outputs",
                        nargs="+")

    parser.add_argument("--margin", "-m",
                        type=float,
                        default=0.0,
                        help="Error margin as percentage (e.g., 0.1 for 10%%)")

    parser.add_argument("--ignore-tags",
                        action="store_true",
                        help="Ignore entity types, only consider spans")

    parser.add_argument("--output-errors",
                        help="Output file for error report")

    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Show detailed per-document results")

    args = parser.parse_args()

    # Validar margen
    if args.margin < 0 or args.margin > 1:
        raise ValueError("Margin must be between 0 and 1 (0% to 100%)")

    print(f"Running evaluation with margin: {args.margin:.2%}")
    print(f"Tags consideration: {'IGNORED' if args.ignore_tags else 'CONSIDERED'}")
    if args.output_errors:
        print(f"Error report will be saved to: {args.output_errors}")
    print()

    evaluate_with_margin(
        args.gs_dir,
        args.sys_dir,
        i2b2Annotation if args.format == "i2b2" else BratAnnotation,
        margin=args.margin,
        ignore_tags=args.ignore_tags,
        output_errors=args.output_errors,
        verbose=args.verbose
    )