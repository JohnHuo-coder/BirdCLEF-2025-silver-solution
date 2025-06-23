import os
import logging
import random
import gc
import math
import warnings
import torchvision
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm.auto import tqdm
import timm
import torchvision
from torch.distributions import Beta
from torch_audiomentations import Compose, PitchShift, Shift
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class CFG:
    
    seed = 42
    debug = False  
    print_freq = 100
    num_workers = 4
    clip_type = 'random'
    OUTPUT_DIR = './'

    train_datadir = '/root/autodl-tmp/birdclef-2025/train_audio'
    train_csv = '/root/autodl-tmp/tmp_data/1train_preprocess_with_human_voice_filter0.5.csv'
    test_soundscapes = '/root/autodl-tmp/BirdCLEF_2025/data/test_soundscapes'
    submission_csv = '/root/autodl-tmp/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/root/autodl-tmp/birdclef-2025/taxonomy.csv'

    
    model_name = 'seresnext26t_32x4d'  
    pretrained = True
    in_channels = 1

    LOAD_DATA = True  
    FS = 32000
    TARGET_DURATION = 5.0
    TRAIN_DURATION = 7.0
    
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 256
    FMIN = 20
    FMAX = 16000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs =  30 
    batch_size = 32  
    criterion = 'FocalLossCE' #'FocalLossCE'#'BCEWithLogitsLoss'
    normal = 80
    n_fold = 5
    selected_folds = [0]   

    optimizer = 'AdamW'
    lr = 5e-4 
    weight_decay = 1e-5
  
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    mixup_prob = 0.1
    mixup_double = 1.0
    mix_beta = 5
    mix_beta2 = 2
    mixup2_prob = 0.7

    mixup = False
    mixup2 = True
    cutmix = False


    

cfg = CFG()



def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)





class BirdCLEFDataset(Dataset):
    def __init__(self, df, cfg, mode="train", transform=None):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.transform = transform

        
        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

            
        if 'filepath' not in self.df.columns:
            self.df['filepath'] = self.cfg.train_datadir + '/' + self.df.filename
        
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        sample_names = set(self.df['samplename'])


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio_sample = self.read_audio(row.filepath)
        audio = torch.tensor(audio_sample[np.newaxis]).float()
  
        target = self.encode_label(row['primary_label'])
        
        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']
            secondary_labels = [label for label in secondary_labels if label in self.label_to_idx]

            if len(secondary_labels)>0: # turn primary label into soft label, takes on 0.5 weight, assign the rest 0.5 uniformly to secondary labels
                target *= 0.5
                sec_weight = 0.5 / len(secondary_labels)
                for label in secondary_labels:
                    target[self.label_to_idx[label]] = sec_weight
        return {
            'audios': audio,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename']
        }
    def read_audio(self, path):
        audio_data, _ = librosa.load(path, sr=self.cfg.FS)
        if self.mode == 'train':
            target_samples = int(self.cfg.TRAIN_DURATION * self.cfg.FS)
        else:
            target_samples = int(self.cfg.TARGET_DURATION * self.cfg.FS)  

        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)
        if self.cfg.clip_type == 'center':
            start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
            end_idx = min(len(audio_data), start_idx + target_samples)
            tmp_audio = audio_data[start_idx:end_idx]
        elif self.cfg.clip_type == 'before':
            end_idx = target_samples
            tmp_audio = audio_data[:end_idx]
        elif self.cfg.clip_type == 'random':
            tmp_audio = audio_data

        chunks_audio = []
        for i in range(math.ceil(len(tmp_audio) / target_samples)):
            audio = tmp_audio[i*target_samples: (i+1)*target_samples]
            if len(audio) < target_samples:
                audio = np.pad(audio, 
                                    (0, target_samples - len(audio)), 
                                    mode='constant')
            chunks_audio.append(audio)
        if self.mode == 'train':
            rand_idx = np.random.choice(list(range(len(chunks_audio))))
        else:
            rand_idx = 0
        
        return chunks_audio[rand_idx]


    
    def encode_label(self, label):
        """Encode label to one-hot vector"""
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target



def collate_fn(batch): 
    """Custom collate function to handle different sized spectrograms"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}
        
    result = {key: [] for key in batch[0].keys()}
    
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    
    for key in result:
        if key == 'target' and isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key])
        elif key == 'audios' and isinstance(result[key][0], torch.Tensor):
            shapes = [t.shape for t in result[key]]
            if len(set(str(s) for s in shapes)) == 1:
                result[key] = torch.stack(result[key])
    
    return result

class Mixup(nn.Module):
    def __init__(self, mix_beta, mixup_prob, mixup_double):
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixup_prob = mixup_prob
        self.mixup_double = mixup_double

    def forward(self, X, Y, weight=None): 
        p = torch.rand((1,))[0] 
        if p < self.mixup_prob: 
            bs = X.shape[0] 
            n_dims = len(X.shape) 
            perm = torch.randperm(bs) 

            p1 = torch.rand((1,))[0] 
            if p1 < self.mixup_double:
                X = X + X[perm]
                Y = Y + Y[perm]
                Y = torch.clamp(Y, 0, 1)

                if weight is None:
                    return X, Y
                else:
                    weight = 0.5 * weight + 0.5 * weight[perm]
                    return X, Y, weight
            else:
                perm2 = torch.randperm(bs)
                X = X + X[perm] + X[perm2]
                Y = Y + Y[perm] + Y[perm2]
                Y = torch.clamp(Y, 0, 1)

                if weight is None:
                    return X, Y
                else:
                    weight = (
                        1 / 3 * weight + 1 / 3 * weight[perm] + 1 / 3 * weight[perm2]
                    )
                    return X, Y, weight
        else:
            if weight is None:
                return X, Y
            else:
                return X, Y, weight

class Mixup2(nn.Module):
    def __init__(self, mix_beta, mixup2_prob):
        super(Mixup2, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta) 
        self.mixup2_prob = mixup2_prob

    def forward(self, X, Y, weight=None):
        p = torch.rand((1,))[0] 
        if p < self.mixup2_prob:
            bs = X.shape[0]
            n_dims = len(X.shape)
            perm = torch.randperm(bs)
            coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device) 

            if n_dims == 2:
                X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm] #[bs,1]
            elif n_dims == 3:
                X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm] #[bs,1,1]
            else:
                X = (
                    coeffs.view(-1, 1, 1, 1) * X
                    + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm] #[bs,1,1,1]
                )
            Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

            if weight is None:
                return X, Y
            else:
                weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
                return X, Y, weight
        else:
            if weight is None:
                return X, Y
            else:
                return X, Y, weight

def sumix(waves: torch.Tensor, labels: torch.Tensor, max_percent: float = 1.0, min_percent: float = 0.3): 
    batch_size = len(labels)
    perm = torch.randperm(batch_size)
    coeffs_1 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (
        max_percent  - min_percent
    ) + min_percent
    coeffs_2 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (
        max_percent  - min_percent
    ) + min_percent
    label_coeffs_1 = torch.where(coeffs_1 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_1)) 
    label_coeffs_2 = torch.where(coeffs_2 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_2)) 
    labels = label_coeffs_1 * labels + label_coeffs_2 * labels[perm]
    waves = coeffs_1 * waves + coeffs_2 * waves[perm]
    return waves, torch.clip(labels, 0, 1)



def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0) # 如果有 bias，就用 0 初始化（推荐做法)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2)) #weight~U(-a,a),a=gain×sqrt(6/(fan_in+fan_out))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02) 
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()

def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled
    
def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)

    return output

class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        ) 
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1) 
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2) 
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)

    
class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.training = True
        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df)

        self.loss_function = get_criterion(cfg)
        self.mixup = Mixup(
            mix_beta=self.cfg.mix_beta,
            mixup_prob=self.cfg.mixup_prob,
            mixup_double=self.cfg.mixup_double,
        )
        self.mixup2 = Mixup2(
            mix_beta=self.cfg.mix_beta2, mixup2_prob=self.cfg.mixup2_prob
        )

        self.time_mask_transform = torchaudio.transforms.TimeMasking( 
            time_mask_param=60, iid_masks=True, p=0.5
        )
        self.freq_mask_transform = torchaudio.transforms.FrequencyMasking( 
            freq_mask_param=24, iid_masks=True
        )
        self.audio_transforms = Compose(
            [
                # AddColoredNoise(p=0.5),
                PitchShift( 
                    min_transpose_semitones=-4,
                    max_transpose_semitones=4,
                    sample_rate=self.cfg.FS,
                    p=0.4,
                ),
                Shift(min_shift=-0.5, max_shift=0.5, p=0.4),
            ]
        )
        self.randaug = torchvision.transforms.RandAugment() 
        self.randerasing = torchvision.transforms.RandomErasing() 
        self.melspec_transform = torchaudio.transforms.MelSpectrogram( 
            sample_rate=self.cfg.FS,
            hop_length=self.cfg.HOP_LENGTH,
            n_mels=self.cfg.N_MELS, 
            f_min=self.cfg.FMIN,
            f_max=self.cfg.FMAX,
            n_fft=self.cfg.N_FFT,
            center=True,
            pad_mode="constant",
            norm="slaney",
            onesided=True,
            mel_scale="slaney",
        )
        # shape: (B,1,T,N_MELS)

        if self.cfg.device == "cuda":
            self.melspec_transform = self.melspec_transform.cuda()
        else:
            self.melspec_transform = self.melspec_transform.cpu()

        self.db_transform = torchaudio.transforms.AmplitudeToDB( 
            stype="power", top_db=80
        )

        self.bn0 = nn.BatchNorm2d(cfg.N_MELS) 

        base_model = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,
            drop_path_rate=0.2
        )
        layers = list(base_model.children())[:-2] 
        self.encoder = nn.Sequential(*layers)

        if "efficientnet" in self.cfg.model_name:
            in_features = base_model.classifier.in_features
        elif "eca" in self.cfg.model_name:
            in_features = base_model.head.fc.in_features
        elif "res" in self.cfg.model_name:
            in_features = base_model.fc.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, cfg.num_classes, activation="sigmoid")
        
        self.init_weight()
        

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def normalize_std(self,spec, eps=1e-6):
        mean = torch.mean(spec)
        std = torch.std(spec)
        return torch.where(std == 0, spec-mean, (spec - mean) / (std+eps)) 
    def extract_feature(self,x):
        x = x.permute((0, 1, 3, 2)) # (B, 1, mel, T) → (B, 1, T, mel)
        frames_num = x.shape[2]

        x = x.transpose(1, 3) # (B, mel, T, 1)
        x = self.bn0(x) # apply batchnorm to mel bin
        x = x.transpose(1, 3) # (B, 1, T, mel)

        x = x.transpose(2, 3) # → (B, 1, mel, T)
        # (batch_size, channels, freq, frames)
        x = self.encoder(x) 

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2) # (B, C, T)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1) 
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)  
        x = x.transpose(1, 2) # → (B, T, C) in order to apply fc to channel dimension
        x = F.relu_(self.fc1(x)) 
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training) 
        return x, frames_num

        
    def forward(self, x,y=None):
        if y is not None: # if y is not none, training mode, apply mixup
            if self.cfg.mixup :
                x, y = self.mixup(x, y) # mixup audio

                
        
        x = self.transform_to_spec(x) # transform audio, then turn it into melspectrogram
        x = self.randerasing(x) 
        if self.cfg.mixup2 and y is not None: # mixup melspectrogram
            x, y = self.mixup2(x, y)
        
        x, frames_num = self.extract_feature(x) # (B, C, T)

        # norm_att: [batch_size, num_classes, time_frames]  
        # segmentwise_output: [batch_size, num_classes, time_frames]
        # clipwise_output: 形状 [batch_size, num_classes]
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x) 
        segmentwise_logit = segmentwise_output.transpose(1, 2) # [batch_size, num_classes, time_frames] → [batch_size, time_frames, num_classes]

        if y is not None: 
            loss = 0.5 * self.loss_function(
                torch.logit(clipwise_output), y
            ) + 0.5 * self.loss_function(segmentwise_logit.max(1)[0], y)
            # Part 1 optimizes classification results after attention weighting (clipwise_output).
            # Part 2 used to prevent weighted averaging (clipwise_output) from diluting strong signals in key frames.

            return torch.logit(clipwise_output), loss.mean() 
        else:
            return torch.logit(clipwise_output)



    # Pink Noise Masking
    def lower_upper_freq(self, images):
        r = torch.randint(self.cfg.N_MELS // 2, self.cfg.N_MELS, size=(1,))[0].item() 
        x = (torch.rand(size=(1,))[0] / 2).item()  
        pink_noise = (
            torch.from_numpy(
                np.array( 
                    [
                        np.concatenate(
                            (
                                1 - np.arange(r) * x / r, 
                                np.zeros(self.cfg.N_MELS - r) - x + 1, 
                            )
                        )
                    ]
                ).T 
            )
            .float()
            .to(self.cfg.device)
        )
        images = images * pink_noise # results in significant suppression of high-frequency regions,while maintaining original intensity in low-frequency components
        return images


    def transform_to_spec(self, audio):
        if self.training:
            audio = self.audio_transforms(audio, sample_rate=self.cfg.FS)

    
        spec = self.melspec_transform(audio)
        spec = self.db_transform(spec)
        if self.cfg.normal == 80: # self.db_transform(spec) -> [-80, 0]
            spec = (spec + 80) / 80 # ->[0,1]
        elif self.cfg.normal == 255: # if melspec was stroed as PNG files, pixel value range ->[0,255] 
            spec = spec / 255
        else:
            raise NotImplementedError

        if self.training:
            spec = self.time_mask_transform(spec)
            if torch.rand(size=(1,))[0] < 0.5:
                spec = self.freq_mask_transform(spec)
            if torch.rand(size=(1,))[0] < 0.5:
                spec = self.lower_upper_freq(spec)
        return spec

class FocalLossBCE(torch.nn.Module): 
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = 'mean',
            bce_weight: float = 1.0,
            focal_weight: float = 1.0, 
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss 


class FocalLossCE(torch.nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
            ce_weight: float = 1.0,
            focal_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(
            reduction=reduction 
        )
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        ce_loss = self.ce(logits, targets)
        return self.ce_weight * ce_loss + self.focal_weight * focall_loss

def get_optimizer(model, cfg):
    
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")
        
    return optimizer

def get_scheduler(optimizer, cfg):
   
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.epochs // 3,  # Restart every 1/3 of total epochs
            T_mult=2,             # Double the period after each restart
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=cfg.min_lr,
            verbose=True
        )
    elif cfg.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs // 3,
            gamma=0.5
        )
    elif cfg.scheduler == 'OneCycleLR':
        scheduler = None  
    else:
        scheduler = None
        
    return scheduler

def get_criterion(cfg):
 
    if cfg.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif cfg.criterion == 'FocalLossBCE':
        criterion = FocalLossBCE(reduction="none")
    elif cfg.criterion == 'FocalLossCE':
        print('FocalLossCE')
        criterion = FocalLossCE()
    elif cfg.criterion == 'CE':
        criterion = nn.CrossEntropyLoss(
                reduction="none"
            )
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")
        
    return criterion



def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.training = True
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    
    for step, batch in pbar:
    
        # if isinstance(batch['audios'], list):
        #     batch_outputs = []
        #     batch_losses = []
        #     for i in range(len(batch['audios'])):
        #         inputs = batch['audios'][i].unsqueeze(0).to(device)
        #         target = batch['target'][i].unsqueeze(0).to(device)
                
        #         optimizer.zero_grad()
        #         output = model(inputs)
        #         loss = criterion(output, target)
        #         loss.backward()
                
        #         batch_outputs.append(output.detach().cpu())
        #         batch_losses.append(loss.item())
            
        #     optimizer.step()
        #     outputs = torch.cat(batch_outputs, dim=0).numpy()
        #     loss = np.mean(batch_losses)
        #     targets = batch['target'].numpy()
            
        # else:
        inputs = batch['audios']
        targets = batch['target']

        optimizer.zero_grad()

        # inputs, targets = mixup(inputs, targets, cfg.mixup_alpha)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs, loss = model(inputs, targets)

        


        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()
            
        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss if isinstance(loss, float) else loss.item())
        
        pbar.set_postfix({
            'train_loss': np.mean(losses[-10:]) if losses else 0,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, auc

def validate(model, loader, criterion, device):
    model.training = False
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            # if isinstance(batch['audios'], list):
            #     batch_outputs = []
            #     batch_losses = []
            #     for i in range(len(batch['audios'])):
            #         inputs = batch['audios'][i].unsqueeze(0).to(device)
            #         target = batch['target'][i].unsqueeze(0).to(device)
                    
            #         output = model(inputs,target)
            #         loss = criterion(output, target)
                    
            #         batch_outputs.append(output.detach().cpu())
            #         batch_losses.append(loss.item())
                
            #     outputs = torch.cat(batch_outputs, dim=0).numpy()
            #     loss = np.mean(batch_losses)
            #     targets = batch['target'].numpy()
                
            # else:
            inputs = batch['audios'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()
            
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            
            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss if isinstance(loss, float) else loss.item())
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, auc

def calculate_auc(targets, outputs):
  
    num_classes = targets.shape[1]
    aucs = []
    
    probs = 1 / (1 + np.exp(-outputs))
    targets = (targets + 0.9).astype(int)
    for i in range(num_classes):
        
        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)
    
    return np.mean(aucs) if aucs else 0.0


def run_training(df, cfg):
    """Training function that can either use pre-computed spectrograms or generate them on-the-fly"""

    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    cfg.num_classes = len(species_ids)
    
    if cfg.debug:
        cfg.update_debug_settings()
        
    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    
    best_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        if fold not in cfg.selected_folds:
            continue
            
        print(f'\n{"="*30} Fold {fold} {"="*30}')
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        print(f'Training set: {len(train_df)} samples')
        print(f'Validation set: {len(val_df)} samples')
        train_df = train_df[train_df['has_human']==False]
        val_df = val_df[val_df['has_human']==False]
        print(f'Training set: {len(train_df)} samples')
        print(f'Validation set: {len(val_df)} samples')

        train_dataset = BirdCLEFDataset(train_df, cfg, mode='train')
        val_dataset = BirdCLEFDataset(val_df, cfg, mode='valid')
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
            generator=torch.Generator().manual_seed(cfg.seed)
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        
        model = BirdCLEFModel(cfg).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)
        
        if cfg.scheduler == 'OneCycleLR':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                steps_per_epoch=len(train_loader),
                epochs=cfg.epochs,
                pct_start=0.1
            )
        else:
            scheduler = get_scheduler(optimizer, cfg)
        
        best_auc = 0
        best_epoch = 0
        
        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")
            
            train_loss, train_auc = train_one_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
            )
            
            val_loss, val_auc = validate(model, val_loader, criterion, cfg.device)

            if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                print(f"New best AUC: {best_auc:.4f} at epoch {best_epoch}")

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'train_auc': train_auc,
                    'cfg': cfg
                }, f"model_fold{fold}.pth")


        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'val_auc': val_auc,
            'train_auc': train_auc,
            'cfg': cfg
        }, f"last_model_fold{fold}.pth")
        best_scores.append(best_auc)
        print(f"\nBest AUC for fold {fold}: {best_auc:.4f} at epoch {best_epoch}")
        
        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*60)
    print("Cross-Validation Results:")
    for fold, score in enumerate(best_scores):
        print(f"Fold {cfg.selected_folds[fold]}: {score:.4f}")
    print(f"Mean AUC: {np.mean(best_scores):.4f}")
    print("="*60)



if __name__ == "__main__":
    import time
    
    print("\nLoading training data...")
    train_df = pd.read_csv(cfg.train_csv)
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)

    print("\nStarting training...")
    print(f"LOAD_DATA is set to {cfg.LOAD_DATA}")
    if cfg.LOAD_DATA:
        print("Using pre-computed mel spectrograms from NPY file")
    else:
        print("Will generate spectrograms on-the-fly during training")
    
    run_training(train_df, cfg)
    
    print("\nTraining complete!")
