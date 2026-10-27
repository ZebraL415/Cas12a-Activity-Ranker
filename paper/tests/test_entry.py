"""Small, data-independent checks for the new entry; numerical assets have their own report."""
import importlib.util
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import reproduce

def test_tf_environment_entry_is_not_dereferenced(monkeypatch,tmp_path):
    runtime=tmp_path/'python-base';runtime.write_text('')
    venv=tmp_path/'tf-venv';venv.mkdir();entry=venv/'python';entry.symlink_to(runtime)
    called=[]
    monkeypatch.setattr(sys,'argv',['reproduce.py','train','--data-dir',str(tmp_path),'--output-dir',str(tmp_path/'out'),
        '--cohort','common_dev','--protocol','target','--outer-fold','0','--tf-python',str(entry)])
    monkeypatch.setattr(reproduce.subprocess,'run',lambda command,**kwargs:called.append(command))
    reproduce.main()
    assert called[0][called[0].index('--tf-python')+1]==str(entry)

def test_archive_path_traversal_is_rejected(tmp_path):
    import io,tarfile
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        info=tarfile.TarInfo('../outside.txt');info.size=1;t.addfile(info,io.BytesIO(b'x'))
    with pytest.raises(tarfile.FilterError): reproduce.unpack(archive,tmp_path/'unpack')

def test_candidate_input_is_required(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'argv',['reproduce.py','predict-candidate','--data-dir',str(tmp_path),'--output',str(tmp_path/'out.csv')])
    with pytest.raises(SystemExit):reproduce.main()
