# BirdCLEF-2025-silver-solution
This repository contains the training code for the Sound Event Detection(SED) model used in the BirdCLEF 2025 Kaggle Competition.

## key Features

### 1. Audio Processing
- Mel spectrogram extraction with configurable parameters
- Audio augmentation pipeline including:  
  - Pitch shifting (±4 semitones)  
  - Time shifting (±50%)  
  - Time and frequency masking  
  - Pink noise masking for high-frequency suppression  

### 2. Model Architecture
- Custom CNN-based model using timm backbone (seresnext26t_32x4d by default)  
- Attention mechanism (AttBlockV2) for weighted feature aggregation  
- Channel smoothing with max+avg pooling  
- Multi-head output (clipwise + segmentwise predictions)  

### 3. Training Features
- Dual-stage mixup augmentation:  
  - Audio-level mixup  
  - Spectrogram-level mixup  
- Stratified K-Fold cross-validation  
- Multiple loss functions supported:  
  - Focal Loss (both BCE and CE variants)  
  - Standard BCEWithLogitsLoss  
- Learning rate scheduling (Cosine, OneCycle, ReduceOnPlateau etc.)  

### 4. Data Handling
- Dynamic audio chunking/padding  
- Secondary label support (soft label encoding)  
- Human voice filtering  
- Custom collate function for variable-length inputs
  
## Dataset
This code uses the BirdCLEF 2025 dataset available [here](https://www.kaggle.com/competitions/birdclef-2025/data)

Run the following to download the dataset: `bash download_data.sh`
## Preprocess
To ensure the preprocessing scripts under the `src` directory work correctly, please run them **from the project root** using the following command:

`python src/preprocess/vad_detection.py`

`python src/preprocess/voice_filter.py`
## How to Train
Run the following code from the project root directory:

`python train.py --config config/config.yaml`
## Inference 
We ensembled our self-trained single SED model with a high-performing public model, drawing inspiration from [this public notebook](https://www.kaggle.com/code/johnyim1570/bird25-weightedblend-nfnet-seresnext-0-878) The final inference, incorporating our model into the ensemble, is implemented in [this notebook](https://www.kaggle.com/code/huojingyu/bird25-weightedblend-nfnet-seresnext-e7d03f)

## License 
This project is licensed under the MIT license 
