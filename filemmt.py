import shutil
import os

######## Settings ########
# Directory where patched-out versions output by remspur.py are located (originally ./out)
start = 'out'
# Where to put modified datasets which are ready to be used in Dyah's code
finish = 'ready'


metadata='./metadata.csv'
readme = './RELEASE_v1.0.txt'

for f in os.listdir(start):
    shutil.copy(metadata, os.path.join(start,f))
    shutil.copy(readme, os.path.join(start,f))

    os.makedirs(os.path.join(finish, f)) # (re)make meaningfully-named enclosing dir in dest folder
    shutil.move(os.path.join(start, f), os.path.join(finish, f)) #move dir
    shutil.move(os.path.join(finish, f, f), os.path.join(finish, f, 'waterbirds_v1.0')) #rename
