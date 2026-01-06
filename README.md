# B2E-CDG: Conditional Diffusion-Based for Label-Free OCT Angiography Artifact Removal and Robust Vascular Reconstruction

B2E-CDG is a conditional diffusion-based framework for label-free OCT angiography (OCTA) artifact removal. The method leverages conditional diffusion models to reconstruct clear, artifact-free normal vascular images from abnormal OCTA images with artifacts.

### Install Dependencies

```bash
pip install -r requirement.txt
```

Main dependencies include:
- `torch>=1.6`
- `torchvision`
- `numpy`
- `opencv-python`
- `pillow`
- `tensorboardx`
- `wandb` (optional)

## 📊 Data Preparation

The dataset should be organized in the following structure:

```
dataroot/
├── abnormal/    # B-scans with artifacts
│   ├── img-0001.png
│   ├── img-0002.png
│   └── ...
└── normal/      # Normal B-scans (for training)
    ├── img-0001.png
    ├── img-0002.png
    └── ...
```

Supported image naming formats:
- `img-XXXX.png`
- `slice_XXXX.png`
- 
## ⚙️ Configuration

The main configuration file is `config/train.json`. Key parameters include:

- **Dataset Configuration**:
  - `dataroot`: Root directory path of the dataset
  - `image_high`, `image_width`: Image dimensions
  - `batch_size`: Batch size
  - `l_resolution`, `r_resolution`: Low and high resolution settings

- **Model Configuration**:
  - `which_model_G`: Model type (`oct` or `ddpm`)
  - `unet`: UNet network structure parameters
  - `beta_schedule`: Noise scheduling for the diffusion process


## 🎯 Usage

### Training

```bash
python train.py -p train -c config/train.json
```

Optional arguments:
- `-p, --phase`: Running phase (`train` or `val`)
- `-c, --config`: Path to configuration file
- `-gpu, --gpu_ids`: Specify GPU IDs (e.g., `0,1,2,3`)
- `-enable_wandb`: Enable WandB logging
- `-log_wandb_ckpt`: Log checkpoints to WandB
- `-log_eval`: Log evaluation results to WandB

### Inference/Testing

```bash
python infer.py -p val -c config/test.json
```

Or use the training script for validation:

```bash
python train.py -p val -c config/train.json
```

### Resume Training

Set `resume_state` in `config/train.json` to the checkpoint path prefix:

```json
"resume_state": "checkpoint/I1000_E10"
```

> Xu J, Fu S, Xing J, et al. B2E-CDG: Conditional diffusion-based for label-free OCT angiography artifact removal and robust vascular reconstruction[J]. Artificial Intelligence in Medicine, 2025: 103345.
