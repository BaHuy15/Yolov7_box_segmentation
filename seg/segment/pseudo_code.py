import os
import glob
from pathlib import Path

path="/home/tonyhuy/bottle_classification/data_bottle_detection/home/pevis/TOMO_detection/data_bottle_detection/test"
files = []
for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
    p = str(Path(p).resolve())
    if '*' in p:
        files.extend(sorted(glob.glob(p, recursive=True)))  # glob
    elif os.path.isdir(p):
        files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
    elif os.path.isfile(p):
        files.append(p)  # files
    else:
        raise FileNotFoundError(f'{p} does not exist')
print(files)
