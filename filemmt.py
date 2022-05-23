import shutil
import os

metadata='/home/sonia/special-octo-pancake/waterbirds_autoLF/metadata.csv'
readme = '/home/sonia/special-octo-pancake/waterbirds_autoLF/RELEASE_v1.0.txt'

start='out'
root = 'ready'

for f in ['th0.7ps75wc7', 'th0.7ps75wc8', 'th0.7ps75wc9', 'th0.7ps75wc11', 'th0.7ps75wc12', 'th0.7ps75wc13', 'th0.7ps75wc14', 'th0.6ps75wc7', 'th0.6ps75wc8', 'th0.6ps75wc9', 'th0.6ps75wc11', 'th0.6ps75wc12', 'th0.6ps75wc13', 'th0.6ps75wc14']:

    if len(os.listdir(os.path.join(start, f))) != 200:
        print('check', f)
    shutil.copy(metadata, os.path.join(start,f))
    shutil.copy(readme, os.path.join(start,f))


    os.mkdir(os.path.join(root, f)) # (re)make meaningfully-named enclosing dir in dest folder
    shutil.move(os.path.join(start, f), os.path.join(root, f)) #move dir
    shutil.move(os.path.join(root, f, f), os.path.join(root, f, 'waterbirds_v1.0')) #rename
