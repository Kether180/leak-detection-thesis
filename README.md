# Deep CNNs for Robotic Leak Detection: A Hybrid Real-Synthetic Data Approach

Master's Thesis - IT University of Copenhagen

**Author:** German Alexander Garcia Angus (gega@itu.dk)  
**Supervisors:** Andres Faina (ITU)
**Co-Supervisor:** Joachim Svendsen (Novo Nordisk)  
**Date:** December 2025

## Overview

This repository contains the code for training ResNet50 classifiers to detect industrial leaks (oil, water, no-leak) using real images, synthetic images generated with Stable Diffusion 2.1, and hybrid combinations.

**Key Results:**
| Experiment | Training Data | Accuracy on Real Images |
|------------|---------------|-------------------------|
| A | Real only (735 images) | 97.47% |
| B | Synthetic only (1,470 images) | 47.47% |
| C | Hybrid (1,470 images) | **99.37%** |

## Repository Structure
```
robot-leak-detection/
├── src/                    # Python scripts
│   ├── generate.py         # Synthetic image generation
│   ├── filter.py           # Quality filtering pipeline
│   ├── train.py            # Model training (clean version)
│   ├── train_resnet50.py   # Model training (robust version)
│   ├── evaluate.py         # Cross-domain evaluation
│   └── gradcam.py          # Grad-CAM visualization
├── configs/                # Experiment configurations
│   ├── exp_a.yaml          # Real-only experiment
│   ├── exp_b.yaml          # Synthetic-only experiment
│   └── exp_c.yaml          # Hybrid experiment
├── data/                   # Datasets (not in git)
├── models/                 # Trained models (not in git)
├── results/                # Evaluation outputs
├── slurm_jobs/             # HPC cluster job scripts
└── logs/                   # Training logs
```

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/kether180/leak-detection-thesis.git
cd leak-detection-thesis
```

### 2. Hardware Requirements

**Training (GPU recommended):**
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1080 (8GB VRAM) | NVIDIA RTX 3090 / A100 |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB SSD | 100 GB SSD |
| CUDA | 11.7+ | 12.0+ |

**Inference (CPU or GPU):**
| Platform | Supported |
|----------|-----------|
| NVIDIA Jetson (Orin, Xavier) | Yes (ONNX + TensorRT) |
| Raspberry Pi 4/5 | Yes (ONNX Runtime) |
| Intel/AMD CPU | Yes (ONNX Runtime) |
| Apple Silicon (M1/M2) | Yes (ONNX Runtime) |

**Synthetic Data Generation (Stable Diffusion):**
- Minimum 12GB VRAM for SD 2.1
- Recommended: NVIDIA A100 / RTX 4090 for batch generation

### 3. Create Environment
```bash
conda create -n leak_detection python=3.10
conda activate leak_detection
pip install -r requirements.txt
```

### 4. Data Sources

The dataset combines multiple sources:

**Real Images:**
- **Roboflow:** Industrial leak detection datasets
- **Pexels API:** Additional real-world leak imagery

**Synthetic Images:**
- Generated using **Stable Diffusion 2.1** with custom prompts
- Quality-filtered using a ResNet18 classifier (40% confidence threshold)

**Dataset Statistics:**
| Class | Real Images | Synthetic Images |
|-------|-------------|------------------|
| Oil Leak | ~245 | ~490 |
| Water Leak | ~245 | ~490 |
| No Leak | ~245 | ~490 |

*Note: Data files are not included in git due to size. Contact the author for access.*

### 5. Trained Models

Pre-trained models (`.pth` files) are excluded from git due to size.
To use the models, either:
1. Train from scratch using the provided scripts
2. Contact the author for pre-trained weights

Place model files in the `models/` directory.

## Usage

### Training

**Option A: Clean training script (as described in thesis)**
```bash
python src/train.py --data_dir data/splits/experiment_a_real_only --output models/model_exp_a.pth --epochs 35
```

**Option B: Robust training script (handles corrupted images)**
```bash
# Edit DATA_DIR in src/train_resnet50.py, then:
python src/train_resnet50.py
```

### Training All Experiments
```bash
# Experiment A: Real-only
python src/train.py --data_dir data/splits/experiment_a_real_only --output models/model_exp_a.pth

# Experiment B: Synthetic-only
python src/train.py --data_dir data/splits/experiment_b_synthetic_only --output models/model_exp_b.pth

# Experiment C: Hybrid
python src/train.py --data_dir data/splits/experiment_c_hybrid --output models/model_exp_c.pth
```

### Synthetic Image Generation
```bash
python src/generate.py --class_name "oil_leak" --num_images 500 --output_dir data/synthetic/oil_leak
```

### Quality Filtering
```bash
python src/filter.py --input_dir data/synthetic --output_dir data/synthetic_filtered --threshold 0.4
```

### Evaluation
```bash
python src/evaluate.py --model models/model_exp_a.pth --test_dir data/splits/experiment_a_real_only/test
```

### Grad-CAM Visualization
```bash
python src/gradcam.py --model models/model_exp_a.pth --image_path test_image.jpg --output gradcam_output.png
```

## HPC Cluster (SLURM)

For ITU HPC cluster users:
```bash
# Start GPU session
./activate_gpu.sh

# Submit training job
sbatch slurm_jobs/job_exp_a.sh
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet50 |
| Pretrained | ImageNet (IMAGENET1K_V2) |
| Learning Rate | 3×10⁻⁴ |
| Batch Size | 32 |
| Epochs | 35 |
| Optimizer | Adam |
| Weight Decay | 10⁻⁴ |
| Random Seed | 42 |

## Key Scripts

| Script | Description |
|--------|-------------|
| `src/train.py` | Main training script (argparse, flexible) |
| `src/train_resnet50.py` | Robust training (handles corrupted images) |
| `src/generate.py` | Generate synthetic images with Stable Diffusion |
| `src/filter.py` | Filter synthetic images by classifier confidence |
| `src/evaluate.py` | Evaluate models on test sets |
| `src/gradcam.py` | Generate Grad-CAM attention visualizations |
| `src/create_experiment_datasets.py` | Create train/val/test splits |
| `src/train_filter_classifier.py` | Train ResNet18 filter classifier |

## Results

### Accuracy Comparison
- Real-only: 97.47%
- Synthetic-only: 47.47% (86% on synthetic test)
- Hybrid: 99.37%

### Domain Gap
Synthetic-trained model drops 38.6 percentage points when tested on real images.

## Future Work

**Improving Synthetic Data Quality:**
- **Domain Randomization:** Vary lighting, camera angles, backgrounds, and textures to reduce synthetic-to-real gap
- **3D Physics Simulation:** Use platforms like NVIDIA Isaac Sim, Blender, or Unity for realistic fluid dynamics and leak behavior
- **Camera/Sensor Simulation:** Add realistic camera noise, lens distortion, and motion blur

**Model Improvements:**
- **YOLO Integration:** Extend from classification to object detection with bounding boxes for leak localization
- **Fine-tuning Pipeline:** Iteratively improve model with new real-world data and re-export to ONNX

**Deployment:**
- **ONNX Export:** Deploy optimized models on edge devices (Jetson, Raspberry Pi)
- **ROS2 Integration:** Real-time leak detection for robotic inspection systems
- **TensorRT Optimization:** Further acceleration for NVIDIA hardware

**MLOps & Continuous Improvement:**
- **Data Drift Monitoring:** Track prediction confidence and flag low-confidence detections
- **Automated Data Collection:** Store edge cases and uncertain predictions for human review
- **Retraining Pipeline:** Trigger retraining when drift metrics exceed thresholds
- **Model Versioning:** Track all model iterations with MLflow or Weights & Biases
- **A/B Testing:** Compare new vs old models before full deployment
- **Continuous ONNX Re-export:** Automated pipeline to convert fine-tuned models back to ONNX
- **Tools:** MLflow, Label Studio, DVC for data versioning, Evidently AI for drift detection

## Citation

If you use this code, please cite:
```bibtex
@mastersthesis{garcia2025leak,
  title={Deep CNNs for Robotic Leak Detection: A Hybrid Real-Synthetic Data Approach},
  author={Garcia Angus, German Alexander},
  school={IT University of Copenhagen},
  year={2025}
}
```

## Contact

German Alexander Garcia Angus - gega@itu.dk

## Running on Your Own System

This code uses **relative paths** and can run anywhere. Just follow these steps:

### Local Machine
```bash
# Clone the repository
git clone https://github.com/kether180/leak-detection-thesis.git
cd leak-detection-thesis

# Install dependencies
pip install -r requirements.txt

# Run scripts from the project root
python src/train.py --data_dir data/splits/experiment_a_real_only --output models/model_exp_a.pth
```

### HPC Cluster (SLURM)
```bash
# Clone to your home directory
cd ~
git clone https://github.com/kether180/leak-detection-thesis.git
cd leak-detection-thesis

# Submit jobs (scripts use ${SLURM_SUBMIT_DIR} for paths)
sbatch slurm_jobs/job_exp_a.sh
```

### Important Notes

1. **Always run from project root:** All scripts expect to be run from the `leak-detection-thesis/` directory

2. **Data not included:** Due to size, data is not in git. Contact the author or regenerate using `src/generate.py`

3. **Models not included:** Train from scratch or contact the author for pre-trained weights

4. **Paths are relative:** The code automatically detects its location - no need to edit paths

### Directory Structure After Setup
```
leak-detection-thesis/
├── data/
│   └── splits/
│       ├── experiment_a_real_only/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── experiment_b_synthetic_only/
│       │   └── ...
│       └── experiment_c_hybrid/
│           └── ...
├── models/
│   ├── model_exp_a.pth
│   ├── model_exp_b.pth
│   └── model_exp_c.pth
└── ... (other files)
```

### Troubleshooting

**"File not found" errors:**
- Make sure you're running from the project root directory
- Check that data is downloaded and in correct location

**"Module not found" errors:**
- Activate your conda environment: `conda activate leak_detection`
- Install requirements: `pip install -r requirements.txt`

**GPU not detected:**
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch GPU: `python -c "import torch; print(torch.cuda.is_available())"`
