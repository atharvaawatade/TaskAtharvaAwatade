# VITS-Based Indian Language Text-to-Speech

## Project Overview
This project implements a VITS (Conditional Variational Autoencoder with Adversarial Learning)-based text-to-speech synthesis system for Indian languages. The implementation focuses on Tamil language (configurable to other Indian languages) using the AI4Bharat data corpus.

## Features
- Complete VITS architecture implementation for Indian languages
- Support for Tamil language (extensible to other Indian languages)
- Multi-speaker support capability
- Mel-spectrogram generation and visualization
- Wandb integration for experiment tracking
- Comprehensive data preprocessing pipeline
- Model checkpointing and evaluation

## Requirements
```
torch==1.13.1
torchaudio==0.13.1
numpy==1.23.5
scipy==1.10.1
soundfile==0.12.1
tensorboard==2.14.0
matplotlib==3.7.1
Unidecode==1.3.6
phonemizer==3.2.1
librosa==0.10.1
wandb==0.15.12
pyworld==0.3.2
```

## Project Structure
```
├── datasets/
│   ├── raw/            # Raw downloaded datasets
│   └── processed/      # Processed dataset splits
├── outputs/
│   ├── checkpoints/    # Model checkpoints
│   └── logs/          # Training logs
├── src/
│   ├── config.py       # Configuration settings
│   ├── dataset.py      # Dataset processing
│   ├── model.py        # VITS model implementation
│   ├── audio.py        # Audio processing utilities
│   └── train.py        # Training script
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vits-indian-tts.git
cd vits-indian-tts
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install system dependencies:
```bash
apt-get update
apt-get install -y build-essential python3-dev
apt-get install -y espeak-ng
apt-get install -y cmake
```

## Usage

### Data Preparation
The system includes an automated data preparation pipeline:
```python
from src.config import VITSConfig
config = VITSConfig()
download_and_prepare_data(config)
```

### Training
To train the model:
```python
# Initialize wandb
wandb.init(project="vits-indic-tts", config=vars(config))

# Create datasets
train_dataset = IndianLanguageDataset(config.processed_path, config, split='train')
val_dataset = IndianLanguageDataset(config.processed_path, config, split='val')

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

# Initialize and train model
model = VITS(config)
train_model(model, train_loader, val_loader, config)
```

### Testing
To test the model and generate samples:
```python
test_dataset = IndianLanguageDataset(config.processed_path, config, split='test')
test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
test_samples = test_model(model, test_loader, config)
```

## Model Configuration
Key configuration parameters can be modified in the `VITSConfig` class:
```python
class VITSConfig:
    def __init__(self):
        self.language = "tamil"  # Change as needed
        self.sampling_rate = 22050
        self.hidden_channels = 192
        self.batch_size = 16
        # ... (other parameters)
```

## Task Coverage Analysis

✅ Completed:
1. Language Selection: Tamil language support implemented
2. Data Preparation:
   - Dataset splitting (80/10/10)
   - Audio processing pipeline
   - Text normalization
3. Model Configuration:
   - VITS architecture implementation
   - Configurable parameters
4. Model Training:
   - Training pipeline
   - Validation process
   - Checkpoint saving
5. Model Evaluation:
   - Test set evaluation
   - Sample generation
   - Visualization

🔄 Partially Completed:
1. Data Acquisition:
   - Currently using LJSpeech dataset
   - Need to integrate AI4Bharat dataset
2. Documentation:
   - Basic documentation provided
   - Need more detailed API documentation

❌ Pending:
1. WAV to FLAC Streaming:
   - WebSocket implementation needed
2. Deployment:
   - Production deployment setup required
3. Advanced Features:
   - Real-time inference
   - API endpoints
