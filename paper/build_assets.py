"""Package four independently usable paper assets; never upload or publish."""
import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

DESCRIPTIONS={
    'inputs':'Processed EasyDesign-derived inputs, formal identities, cohort rosters and the released EasyDesign architecture. Reused data retain upstream attribution. The whole publisher workbook is obtained from the source article.',
    'results':'Accepted and recalculated paper results: fixed/nested predictions, classic baselines, same-capacity blocks, controlled perturbations with their own intact references, historical continuity, figure sources and S6.',
    'nested':'Detailed nested validation: 160 inner fits and 40 outer fits, component predictions, fit/query records, candidate scores and selected rules. The saved inner component CSVs must be read with float_precision=round_trip.',
    'candidate':'Additional TriCasRank candidate refitted on all 8,417 development records. This archive includes its standalone loader and 27 regression fixtures; it does not replace the default repository model.',
}

def include_file(relative):
    name=str(relative)
    # Keep one authoritative prediction table per analysis. Components and
    # candidate scores remain per fold so nested decisions can be reconstructed.
    if name.startswith('results/controls/predictions/'):return False
    if name.startswith('results/subgroups/historical_predictions/'):return False
    if name=='results/accepted/nested/nested_candidate_scores.csv':return False
    if name.startswith('nested/runs/') and name.endswith('/predictions.csv'):return False
    return True

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--repository-commit',required=True)
    parser.add_argument('--stage',choices=['review','published'],default='review')
    parser.add_argument('--tag',default='paper-2026-10-27')
    args=parser.parse_args();data=args.data_dir.resolve();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    release={'repository':'ZebraL415/Cas12a-Activity-Ranker','tag':args.tag,'status':args.stage,'repository_commit':args.repository_commit,'assets':[]}
    for name,description in DESCRIPTIONS.items():
        root=data/name
        if not root.is_dir():raise FileNotFoundError(root)
        (root/'ASSET_README.md').write_text(f'# TriCasRank paper: {name}\n\n{description}\n\nExtract all selected archives into the same directory. See the matching Git repository paper/README.md, result_index.csv and DATA_DICTIONARY.md. All paths are relative to that extracted directory.\n\nProject code: Apache-2.0. Derived research artifacts: CC BY 4.0 with existing upstream attributions. EasyDesign source: Huang et al. (2024), iMeta 3:e214, https://doi.org/10.1002/imt2.214 and https://github.com/scRNA-Compt/EasyDesign.\n')
        (root/'asset_metadata.json').write_text(json.dumps({'asset':name,'description':description,'repository':release['repository'],
            'repository_commit':args.repository_commit,'release_tag':args.tag,'status':args.stage,'cohorts':{'core_dev':8417,'common_dev':5995,'historical_fixed':2217},
            'label':'source-published processed 30-minute activity; higher is more active',
            'model_public_name':'TriCasRank','default_model_replaced':False,'documentation':'paper/DATA_DICTIONARY.md'},indent=2)+'\n')
        paths=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='FILES.csv' and '__pycache__' not in p.parts and p.suffix!='.pyc' and include_file(p.relative_to(data)))
        with (root/'FILES.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=['file','bytes','sha256']);writer.writeheader()
            for p in paths:writer.writerow({'file':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest(p)})
        archive=output/f'tricasrank-paper-{name}.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=6) as target:
            for p in [*paths,root/'FILES.csv']:
                target.add(p,arcname=str(p.relative_to(data)),recursive=False)
        with tarfile.open(archive) as target:
            for member in target.getmembers():
                if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                    raise ValueError(f'Invalid archive entry: {member.name}')
        release['assets'].append({'name':archive.name,'directory':name,'bytes':archive.stat().st_size,'sha256':digest(archive),'file_count':len(paths)+1})
        print(f'Packaged {archive.name}: {archive.stat().st_size/1024**2:.1f} MiB, {len(paths)+1} files',flush=True)
    (output/'release_manifest.json').write_text(json.dumps(release,indent=2)+'\n')

if __name__=='__main__':main()
