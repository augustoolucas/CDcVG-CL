import os

for file in os.walk('./'):
    if 'params' in file[0]:
        os.system(f'cp {file[0]}/CVAE_Model {file[0]}/Specific_Model')
