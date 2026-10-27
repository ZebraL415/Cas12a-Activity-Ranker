"""Location-independent paths shared by the public paper commands."""
from pathlib import Path
import json

PAPER = Path(__file__).resolve().parent
PROTOCOLS = ('target', 'guide', 'atomic_template', 'dual_atomic')
COHORTS = ('core_dev', 'common_dev')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def settings(data_root):
    root = Path(data_root).resolve()
    config = read_json(PAPER / 'config/run.json')
    paths = config['source_paths']
    paths['project_root'] = str(root / 'inputs')
    for key in ('data', 'assignments', 'sequence_manifest', 'template_reference', 'easydesign_contexts', 'old_target_predictions', 'weight_audit'):
        paths[key] = str(root / 'inputs' / paths[key])
    paths['easydesign_config'] = str(PAPER / 'config/easydesign_vectorized_retraining.json')
    paths['easydesign_model'] = str(root / 'inputs/easydesign_model')
    paths['easydesign_vectorization'] = str(PAPER / 'engines/easydesign/easydesign_vectorized.py')
    paths['easydesign_runner'] = str(PAPER / 'engines/easydesign/run_easydesign.py')
    return config
