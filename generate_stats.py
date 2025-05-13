import os
import sys
from collections import defaultdict
from lxml import etree
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter


def compare_annotations(gold_dir, system_dir, output_file):
    """Compare annotations and generate detailed error report"""
    tag_errors = defaultdict(lambda: {'fp': [], 'fn': [], 'tp': []})
    tag_sentences = defaultdict(int)

    with open(output_file, 'w') as out_f:
        out_f.write("MEDDOCAN Error Analysis Report\n")
        out_f.write("=" * 50 + "\n\n")

        for filename in os.listdir(gold_dir):
            if filename.endswith(".xml"):
                gold_file = os.path.join(gold_dir, filename)
                system_file = os.path.join(system_dir, filename)

                if not os.path.exists(system_file):
                    continue

                # Parse gold annotations
                gold_ann = parse_i2b2_annotations(gold_file)
                # Parse system annotations
                sys_ann = parse_i2b2_annotations(system_file)

                out_f.write(f"\nDocument: {filename}\n")
                out_f.write("-" * 50 + "\n")

                # Count sentences for this document
                num_sentences = gold_ann['text'].count('.') + gold_ann['text'].count('\n')
                for ann in gold_ann['annotations']:
                    tag_type = ann.get('subtype', ann['type'])
                    tag_sentences[tag_type] += num_sentences

                # Compare annotations
                compare_document_annotations(gold_ann, sys_ann, tag_errors, out_f)

        # Generate summary by tag type
        out_f.write("\n\nSummary by Tag Type:\n")
        out_f.write("=" * 50 + "\n")
        for tag_type, errors in tag_errors.items():
            out_f.write(f"\nTag Type: {tag_type}\n")
            out_f.write(f"True Positives: {len(errors['tp'])}\n")
            out_f.write(f"False Positives: {len(errors['fp'])}\n")
            out_f.write("Examples:\n")
            for fp in errors['fp'][:3]:
                out_f.write(f"- FP: {fp['text']} (System: {fp['system_text']})\n")
            out_f.write(f"False Negatives: {len(errors['fn'])}\n")
            for fn in errors['fn'][:3]:
                out_f.write(f"- FN: {fn['text']} (Expected: {fn['gold_text']})\n")
            out_f.write("-" * 30 + "\n")

        compute_and_save_metrics(tag_errors, tag_sentences, os.path.dirname(output_file))


def parse_i2b2_annotations(xml_file):
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


def compare_document_annotations(gold, system, tag_errors, out_f):
    gold_ann = gold['annotations']
    sys_ann = system['annotations']
    false_positives = []
    false_negatives = []

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

    if false_positives or false_negatives:
        out_f.write("Errors found:\n")
        if false_positives:
            out_f.write("\nFalse Positives (System tagged but shouldn't):\n")
            for fp in false_positives:
                out_f.write(f"- {fp.get('subtype', fp['type'])}: '{fp['text']}' "
                            f"(pos: {fp['start']}-{fp['end']})\n")
                out_f.write(f"  Context: '{get_context(system['text'], fp['start'], fp['end'])}'\n")

        if false_negatives:
            out_f.write("\nFalse Negatives (System missed):\n")
            for fn in false_negatives:
                out_f.write(f"- {fn.get('subtype', fn['type'])}: '{fn['text']}' "
                            f"(pos: {fn['start']}-{fn['end']})\n")
                out_f.write(f"  Context: '{get_context(gold['text'], fn['start'], fn['end'])}'\n")
    else:
        out_f.write("No errors found in this document.\n")


def get_context(text, start, end, window=30):
    context_start = max(0, start - window)
    context_end = min(len(text), end + window)
    before = text[context_start:start]
    annotated = text[start:end]
    after = text[end:context_end]
    return f"...{before}>>>{annotated}<<<{after}..."


def leak_score(fn, num_sentences):
    try:
        return round(len(fn) / num_sentences, 4) if num_sentences > 0 else 0.0
    except Exception:
        return "NA"


def compute_and_save_metrics(tag_errors, tag_sentences, output_dir):
    tags, precisions, recalls, f1s, leaks = [], [], [], [], []

    for tag, values in tag_errors.items():
        tp = len(values['tp'])
        fp = len(values['fp'])
        fn = len(values['fn'])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        leak = leak_score(values['fn'], tag_sentences.get(tag, 0))

        tags.append(tag)
        precisions.append(round(precision, 4))
        recalls.append(round(recall, 4))
        f1s.append(round(f1, 4))
        leaks.append(leak)

    metrics_df = pd.DataFrame({
        'Tag': tags,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1s,
        'Leak Score': leaks
    })

    excel_path = os.path.join(output_dir, "metrics_by_tag.xlsx")
    metrics_df.to_excel(excel_path, index=False)

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
    print(f"Guardado en: {excel_path}")


def main():
    if len(sys.argv) != 4:
        print("Usage: python error_analysis.py gold_dir system_dir output_file")
        sys.exit(1)

    gold_dir = sys.argv[1]
    system_dir = sys.argv[2]
    output_file = sys.argv[3]

    compare_annotations(gold_dir, system_dir, output_file)
    print(f"Guardado en: {output_file}")


if __name__ == "__main__":
    main()
