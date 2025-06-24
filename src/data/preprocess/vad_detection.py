### Reference: https://www.kaggle.com/code/kdmitrie/bc25-separation-voice-from-data/notebook

import torch
from glob import glob
import pickle

torch.set_num_threads(1)
model, (get_speech_timestamps, _, read_audio, _, _) = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

files = sorted(glob('/kaggle/input/birdclef-2025/train_audio/*/*.ogg'))
voice_data = {}
for fname in files:
    wav = read_audio(fname)
    speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True, threshold=0.4) # 0.4 yields best performance
    if len(speech_timestamps):
        voice_data[fname] = speech_timestamps

        with open('train_voice_summary.txt', 'a') as f:
            f.write(f'{fname}\n')
            
with open('train_voice_data_0.4.pkl', 'wb') as f:
    pickle.dump(voice_data, f)
