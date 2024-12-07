import subprocess
import os

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '1'

for dataset_id in [386, 387, 392, 393, 394, 390]:
    dataset = f'zjumocap_{dataset_id}_mono'
    for mode in ['train', 'test', 'predict']:
        cmd = ['python']
        if mode == 'train':
            # CUDA_VISIBLE_DEVICES=1 python train.py dataset=zjumocap_377_mono
            cmd.extend(['train.py',  f'dataset={dataset}'])
        elif mode == 'test':
            # CUDA_VISIBLE_DEVICES=1 python render.py mode=test dataset.test_mode=view dataset=zjumocap_377_mono
            cmd.extend(['render.py', f'mode={mode}', f'dataset.test_mode=video', f'dataset={dataset}'])
        else:
            # python render.py mode=predict dataset.predict_seq=0 dataset=zjumocap_377_mono
            cmd.extend(['render.py', f'mode={mode}', f'dataset.predict_seq=0', f'dataset={dataset}'])
        subprocess.call(cmd,env=env)