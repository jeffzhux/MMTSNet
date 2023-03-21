import os
import shutil

source_img = 'F:/YOLOP/data/BDD100K/images'
source_json = 'F:/YOLOP/data/AIDEA/train_seg'
target_json = 'F:/YOLOP/data/BDD100K/labels_v2'

# # train
for root, dirs, file in os.walk(f'{source_img}/train'):
    for f in file:
        f = f.replace('.jpg','.json')
        shutil.copy(f'{source_json}/{f}', f'{target_json}/train')
# valid
for root, dirs, file in os.walk(f'{source_img}/valid'):
    for f in file:
        f = f.replace('.jpg','.json')
        shutil.copy(f'{source_json}/{f}', f'{target_json}/valid')