"""Deterministic local inventory; dataset licenses never propagate to code."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('additive_heat', 'mendeley_heat')
URL = 'https://data.mendeley.com/datasets/sbcf36976g/1'
DOI = '10.17632/sbcf36976g.1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asset_id(path):
    return 'am:' + Path(path).resolve().relative_to(ROOT).as_posix()


def evidence(path, quote=None):
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    quote = text if quote is None else quote
    start = text.index(quote)
    return {'asset_id': asset_id(path), 'source_id': asset_id(path), 'quote': quote,
            'locator': {'path': path.relative_to(ROOT).as_posix(),
                        'line': text[:start].count('\n') + 1,
                        'start_char': start, 'end_char': start + len(quote)}}


def build_catalog():
    files, assignments = set(), {}
    for dataset in DATASETS:
        base = ROOT / 'pde_discovery/data' / dataset
        files.update(p for p in base.rglob('*') if p.is_file())
        for path in base.glob('*/manifest.json'):
            manifest = json.loads(path.read_text(encoding='utf-8'))
            entries = list(manifest['splits'].items())
            entries += [('final', e) for e in manifest.get('holdouts', [])]
            for role, entry in entries:
                raw = (path.parent / entry['file']).resolve()
                if not raw.is_relative_to(base.resolve()):
                    raise ValueError('Manifest path escapes dataset')
                if sha(raw) != entry['sha256']:
                    raise ValueError(f'Manifest hash mismatch: {raw.name}')
                assignments[raw] = (path.parent.name, role, path, entry)
    for folder in ('additive_heat', 'refined_heat', 'mendeley_heat'):
        files.update((ROOT / 'artifacts' / folder).rglob('*.json'))
    for name in ('ADDITIVE_REPORT', 'MENDELEY_REPORT', 'UNUSED_DATA_METADATA',
                 'ADDITIVE_PROTOCOL', 'REFINEMENT_PROTOCOL'):
        files.add(ROOT / 'experiments' / (name + '.md'))
    cards = []
    for path in sorted(files):
        suffix = path.suffix.lower()
        modality = {'.csv':'csv', '.xlsx':'xlsx', '.pdf':'pdf', '.py':'code',
                    '.png':'image', '.md':'markdown', '.json':'json'}.get(suffix, 'text')
        author_code = path.name.startswith('author_')
        role = ('raw_measurement' if suffix == '.csv' else 'method_code' if author_code
                else 'protocol' if path.name == 'manifest.json' or 'PROTOCOL' in path.name
                else 'result_table' if 'artifacts' in path.parts else 'report')
        card = dict(asset_id=asset_id(path), title=path.name, path=path.relative_to(ROOT).as_posix(),
                    modality='code' if author_code else modality, domain_role=role,
                    source_url=URL, doi=DOI, license='待核实（本地说明/派生结果）',
                    sha256=sha(path), byte_size=path.stat().st_size, rows=None, cols=None,
                    linked_entities=[], usage_status='metadata_or_result', origin='自制/派生说明')
        if author_code:
            card.update(source_url='https://github.com/lbtcn6/oneDkhEstimator/blob/V1.0.0/oneDkhEstimator.py',
                        license='待核实（不得沿用数据集许可）', origin='作者源码留档')
        if path.resolve() in assignments:
            group, split, manifest_path, entry = assignments[path.resolve()]
            card.update(license='CC BY 4.0', origin='公开真实测量',
                        attribution='Lauren Tomanek; Daniel Stutts', linked_entities=[group, split],
                        usage_status='evaluated_not_blind' if split == 'final' else 'used_for_modeling',
                        manifest_sha256=entry['sha256'], evidence=[evidence(manifest_path)])
        if suffix == '.csv':
            with path.open(encoding='utf-8', newline='') as stream:
                reader = csv.reader(stream)
                header = next(reader)
                card.update(cols=len(header), rows=sum(1 for row in reader if row), header=header)
            if path.resolve() not in assignments:
                card['usage_status'] = 'unknown'  # absence of a manifest is not proof of non-use
        cards.append(card)
    counts = Counter(c['usage_status'] for c in cards if c['modality'] == 'csv')
    return {'schema_version': 1, 'assets': cards,
            'dormancy': {key: counts[key] for key in ('used_for_modeling', 'never_used', 'evaluated_not_blind', 'unknown')},
            'usage_note': '以现有报告为准：历史 UNUSED 元数据不是当前未用证明；未下载的归档成员不计作本地资产。'}


def main():
    result = build_catalog()
    Path(__file__).with_name('asset_catalog.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['dormancy'], ensure_ascii=False))


if __name__ == '__main__':
    main()
