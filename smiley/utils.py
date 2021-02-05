from os.path import abspath
from pathlib import Path

class LoadError(Exception):
    """Raised when the data is not loaded"""
    pass



class Paths:
    src_dirname = 'smiley'
    root_dir = Path(abspath(__file__)).parents[1]
    references_dirname = 'references'
    data_dirname = 'data'
    app_dirname = 'app'
    model_dirname = 'model'
    tests_dirname = 'tests'
    mlruns_dirname = 'mlruns'

    def __init__(self, force=False):
        if not self.src_dirname in self.root_dir.iterdir():
            self.root_dir = self.root_dir.parents[0]
        self.d = {
            'src': self.root_dir / self.src_dirname,
            'data': self.root_dir / self.data_dirname,
            'model': self.root_dir / self.model_dirname,
            # 'mlruns': self.root_dir / self.mlruns_dirname
        }
        self.dirs = list(self.d.keys())
        #for d in self.d:
        #    if not self.d.get(d).exists():
        #        self.d.get(d).mkdir()
        
    def __getattr__(self, name):
        if name in self.d:
            return self.d.get(name)
        else:
            raise AttributeError(f"Attribute {name} does not exists (existing: {self.dirs}).")

    def __getitem__(self, name):
        return getattr(self, name)
