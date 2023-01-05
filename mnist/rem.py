import os
import shutil

folder = 'saved-models/pl/lightning_logs/'

contents = os.listdir(folder)

for cont in contents:
    if os.path.isdir(os.path.join(folder, cont)) and cont != 'version_11':
        shutil.rmtree(os.path.join(folder, cont))