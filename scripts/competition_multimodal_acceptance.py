"""Record real-file acceptance separately from manufactured fixture tests."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from competition_agent.multimodal import ROOT, MissingDependency, parse_asset, project, conflicts

def main():
    output=ROOT/'competition_agent/build/T03';output.mkdir(parents=True,exist_ok=True)
    raw=ROOT/'competition_agent/assets_raw'
    requests=[(raw/'dib-2021-107632.pdf','pdf')]+[(raw/name,'xlsx') for name in ['AdditiveRodResults.xlsx','Temperature Results.xlsx','Wind Results.xlsx']]
    requests += [(ROOT/'pde_discovery/data/mendeley_heat/author_oneDkhEstimator_V1.0.0_full_source.txt','code')]
    requests += [(raw/'examples'/f'自制示例-{group}.png','image') for group in ['xn','xp']]
    result={'task':'T03','real_file_checks':[],'complete':False,'vlm':'optional, disabled; no network calls'}
    for path,kind in requests:
        item={'path':path.relative_to(ROOT).as_posix(),'kind':kind}
        try:
            if not path.exists(): raise FileNotFoundError(path)
            item['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            asset=parse_asset(path,kind,output/'thumbnails')
            (output/(path.stem+'.parsed.json')).write_text(json.dumps(asset,ensure_ascii=False,indent=2),encoding='utf-8')
            item.update(status='parsed',blocks=len(asset['blocks']))
            if kind=='code':
                item['geometry']=[b for b in asset['blocks'] if 225<=b['locator']['line']<=245]
            elif kind=='pdf':
                words=('porous','direction','five','5 times')
                item['key_sentence_candidates']=[b for b in asset['blocks'] if b['type']=='text' and any(w in b['text'].lower() for w in words)]
                item['semantic_acceptance']='manual review of direction/porosity/repeats and page+bbox required'
            elif kind=='image':
                document,spans=project(asset)
                record={'asset':asset,'document':document,'spans':spans}
                metadata=ROOT/'pde_discovery/data/additive_heat/metadata_verification.md'
                lines=metadata.read_text(encoding='utf-8').splitlines()
                graph=[]
                for group in ['xn','xp']:
                    line=next((i,t) for i,t in enumerate(lines,1) if t.startswith('| '+group+' |'))
                    graph.append({'subject':group,'predicate':'has_build_direction','object':'x',
                        'source_id':hashlib.sha256(metadata.read_bytes()).hexdigest(),'quote':line[1],
                        'locator':{'file':metadata.relative_to(ROOT).as_posix(),'line':line[0]},
                        'origin':'public metadata reference for acceptance; not an existing case graph'})
                item['conflicts']=conflicts({'record_card':record},graph)
                item['expected_conflicts']=1 if 'xn' in path.stem else 0
                item['conflict_check_passed']=len(item['conflicts'])==item['expected_conflicts']
        except (MissingDependency,FileNotFoundError) as exc:
            item.update(status='pending',reason=str(exc))
        result['real_file_checks'].append(item)
    (output/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:v for k,v in x.items() if k in ('path','status','reason','blocks')} for x in result['real_file_checks']],ensure_ascii=False,indent=2))

if __name__=='__main__': main()
