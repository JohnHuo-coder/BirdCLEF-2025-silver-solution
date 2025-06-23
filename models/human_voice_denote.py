import os
import pickle
import pandas as pd
import librosa

with open('train_voice_data_0.4.pkl', 'rb') as f:
    voice_data = pickle.load(f)

data_dir = './birdclef-2025/train_audio'

qualified_voice_filenames = set()

for full_path, timestamps in voice_data.items():
    total_voice_duration = sum(item['end'] - item['start'] for item in timestamps)
    filename = '/'.join(full_path.split('/')[-2:])
    filename_path = os.path.join(data_dir, filename)
    try:
        audio_duration = librosa.get_duration(filename=filename_path)
    except Exception as e:
        print(f"Failed to load {filename_path}: {e}")
        continue
    if total_voice_duration / audio_duration > 0.5:
        qualified_voice_filenames.add(filename)

df = pd.read_csv('train_preprocess.csv')
df['has_human'] = df['filename'].isin(qualified_voice_filenames)

print(f"number of samples with more than 50% human voice: {df[df['has_human'] == True].shape[0]}")
df.to_csv('train_preprocess_with_human_voice_filter0.5.csv', index=False)