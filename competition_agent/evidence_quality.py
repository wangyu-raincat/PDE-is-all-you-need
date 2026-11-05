"""Deterministic evidence retrieval and graph checks; no semantic-truth claims."""
from collections import Counter, deque
import math
import re
import unicodedata
from competition_agent import preprocess as inputs

REQUIREMENTS = inputs.REQUIREMENTS
ALIASES = ({'单位', 'unit', 'units', 'uses_unit'},
           {'边界', 'boundary', 'has_boundary'},
           {'字段', 'field', 'records_field'},
           {'测点', 'sensor', 'sensors', 'measures'},
           {'周期', 'periodic'}, {'无量', '量纲', 'dimensionless'})
TOKEN = re.compile(r'[\u3400-\u9fff]+|[^\W\u3400-\u9fff]+', re.UNICODE)


def tokens(text):
    result = []
    for match in TOKEN.finditer(text):
        word = unicodedata.normalize('NFKC', match.group()).casefold()
        if re.fullmatch(r'[\u3400-\u9fff]+', word):
            result.extend(word[i:i+2] for i in range(len(word)-1))
            if len(word) == 1:
                result.append(word)
        else:
            result.append(word)
    return result


def span(document, quote):
    start = document['text'].find(quote)
    offset = 0
    for line in document['text'].splitlines(keepends=True):
        if line.strip() == quote.strip() and quote in line:
            start = offset + line.index(quote)
            break
        offset += len(line)
    if start < 0:
        return None
    end = start + len(quote)
    return {'start_char': start, 'end_char': end,
            'line': document['text'].count('\n', 0, start) + 1,
            'end_line': document['text'].count('\n', 0, max(start, end-1)) + 1,
            'offset_unit': 'Unicode code points, half-open'}


def assertion(fact, documents, prepared_record=None):
    """Recognize a whole-line literal triple, NOT a substring of a negation."""
    doc = documents.get(fact['source_id'])
    if not doc:return False
    doc={'title':doc.get('title','evidence'),'kind':doc.get('kind','text'),'text':doc['text']}
    parsed=prepared_record if prepared_record is not None else inputs.parse_document(doc,fact['source_id'])
    if parsed['status']=='blocked':return False
    if inputs.matching_fact(fact,{fact['source_id']:doc},parsed) is not None:
        return True
    literal = f"{fact['subject']} {fact['predicate']} {fact['object']}"
    return fact['quote'].strip() in (literal, literal + '.') and any(
        line.strip() == fact['quote'].strip() for line in doc['text'].splitlines())


def retrieve(documents, query, limit):
    terms = set(tokens(query))
    expanded = set(terms)
    for group in ALIASES:
        if terms & group:
            expanded.update(group)
    rows = []
    for source, doc in documents.items():
        offset = 0
        for number, line in enumerate(doc['text'].splitlines(keepends=True), 1):
            content = line.rstrip('\r\n')
            rows.append((source, doc, number, offset, content, Counter(tokens(content))))
            offset += len(line)
    if not rows or not expanded:
        return {'method': 'lexical BM25 with Chinese bigrams and explicit bilingual aliases; no embeddings or LLM', 'hits': []}
    count = len(rows)
    average = sum(sum(row[5].values()) for row in rows) / count or 1
    df = Counter(t for row in rows for t in expanded if t in row[5])
    hits = []
    for source, doc, number, offset, line, frequencies in rows:
        length = sum(frequencies.values())
        matches = expanded & frequencies.keys()
        score = sum((1 if t in terms else .55) * math.log(1+(count-df[t]+.5)/(df[t]+.5)) *
                    frequencies[t]*2.2/(frequencies[t]+1.2*(.25+.75*length/average)) for t in matches)
        if score <= 0:
            continue
        # Find the matching token in the original line, so Unicode normalization
        # never invalidates citation offsets and late matches survive truncation.
        positions = []
        for m in TOKEN.finditer(line):
            if re.fullmatch(r'[\u3400-\u9fff]+', m.group()):
                positions.extend(m.start()+i for i in range(len(m.group())-1) if m.group()[i:i+2] in matches)
            elif set(tokens(m.group())) & matches:
                positions.append(m.start())
        anchor = positions[0] if positions else 0
        start = max(0, min(anchor-200, len(line)-2000))
        quote = line[start:start+2000]
        hits.append({'source_id': source, 'source_sha256': source, 'source_hash_scope': 'canonical title/kind/text record', 'title': doc['title'],
                     'line': number, 'quote': quote, 'start_char': offset+start,
                     'end_char': offset+start+len(quote), 'offset_unit': 'Unicode code points, half-open',
                     'matched_terms': sorted(matches), 'score': score})
    return {'method': 'lexical BM25 with Chinese bigrams and explicit bilingual aliases; no embeddings or LLM',
            'hits': sorted(hits, key=lambda h: (-h['score'], h['source_id'], h['start_char']))[:limit]}


def assess(state, conflicts):
    facts, docs = state['facts'], state['documents']
    prepared=state.get('input_preparation') or inputs.prepare(list(docs.values()))
    asserted = [f for f in facts if assertion(f, docs, prepared['records'].get(f['source_id']))]
    covered = {(f['subject'], f['predicate'], f['object']) for f in asserted}
    missing = [dict(zip(('subject', 'predicate', 'object'), r)) for r in REQUIREMENTS if r not in covered]
    asserted_ids={f['id'] for f in asserted}
    unsupported = [f['id'] for f in facts if f['id'] not in asserted_ids]
    mismatches = [f['id'] for f in facts for s, p, o in REQUIREMENTS
                  if (f['subject'], f['predicate']) == (s, p) and f['object'] != o]
    pending = [p['id'] for p in state['ontology_proposals'] if p['status'] == 'pending']
    blockers = [{'code':'input_preparation_blocked','details':[i for i in prepared['summary']['issues'] if i['priority']=='P0']}] if prepared['summary']['blocked'] else []
    for code, values in [('missing_required_metadata', missing), ('unverified_relation_assertions', unsupported),
                         ('metadata_disagrees_with_simulator', mismatches), ('conflicting_sources', conflicts),
                         ('pending_ontology_review', pending)]:
        if values:
            blockers.append({'code': code, 'details': values})
    return {'metadata_ready': not blockers, 'blockers': blockers, 'required_metadata':
            [dict(zip(('subject', 'predicate', 'object'), r)) for r in REQUIREMENTS],
            'asserted_facts': len(asserted), 'total_facts': len(facts),
            'scope': 'fixed synthetic observation metadata consistency only',
            'semantic_truth_verified': False, 'industrial_readiness': False}


def paths(state, source, target, max_hops=4, limit=3):
    """Bounded directed evidence paths; every edge remains a cited assertion."""
    prepared=state.get('input_preparation') or inputs.prepare(list(state['documents'].values()))
    adjacency = {}
    for fact in state['facts']:
        adjacency.setdefault(fact['subject'], []).append(fact)
    for edges in adjacency.values():
        edges.sort(key=lambda f: f['id'])
    queue = deque([(source, [], {source})])
    found, expanded = [], 0
    while queue and len(found) < limit and expanded < 2000:
        node, edges, visited = queue.popleft()
        expanded += 1
        if len(edges) >= max_hops:
            continue
        for fact in adjacency.get(node, []):
            nxt = fact['object']
            if nxt in visited:
                continue
            trail = edges + [fact]
            if nxt == target:
                found.append({'edges': trail, 'source_ids': sorted({f['source_id'] for f in trail}),
                              'all_relations_asserted': all(assertion(f, state['documents'],prepared['records'].get(f['source_id'])) for f in trail)})
                if len(found) == limit:
                    break
            else:
                queue.append((nxt, trail, visited | {nxt}))
    return {'paths': found, 'found': bool(found), 'search_truncated': bool(queue) or len(found) == limit,
            'max_hops': max_hops, 'expanded_nodes': expanded, 'new_facts_created': False,
            'semantic_truth_verified': False, 'scope': 'paths over cited assertions; not inferred physical causation'}
