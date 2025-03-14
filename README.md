# Fashion Image Retrieval with Supervised Contrastive Learning

> A100 GPU-optimized implementation for visual similarity search between worn fashion items and product images

## 📋 Overview

This project implements a state-of-the-art fashion image retrieval system using Supervised Contrastive Learning (SupCon). The system learns to align embeddings between **wearing images** (people wearing fashion items) and **product images** (catalog photos of fashion items), enabling accurate retrieval of products from worn item queries.

### Key Features

-   **A100 GPU Optimization**: Fully optimized for NVIDIA A100 80GB GPUs with high-performance configurations
-   **Memory Bank**: Enhanced contrastive learning using a large memory bank of embeddings (16K+)
-   **Detailed Monitoring**: Comprehensive WandB integration for real-time training visualization
-   **ConvNeXt Backbone**: State-of-the-art CNN architecture with various scaling options
-   **Adaptive Hard Mining**: Focuses learning on the most challenging negative samples

## 🚀 Performance Optimizations

The codebase is specifically optimized for high-performance training on A100 GPUs:

-   **Large Batch Training**: Supports batch sizes of 512+ on A100 80GB GPUs
-   **Mixed Precision**: Automatic FP16 training for increased throughput
-   **Memory Bank**: 16K+ embedding memory bank for effective contrastive learning
-   **Gradient Accumulation**: Optional accumulation for effectively larger batches
-   **Gradient Checkpointing**: Memory-efficient backpropagation for larger models
-   **LARS Optimizer**: Layer-adaptive learning rate scaling for large-batch training
-   **Efficient Data Loading**: Optimized data loading pipeline with configurable workers

## 🛠️ Requirements

-   NVIDIA A100 GPU (80GB recommended)
-   CUDA 11.0+
-   8+ vCPUs
-   125GB+ RAM

### Python Dependencies

```bash
# Install dependencies
pip install -r requirements.txt
```

Key dependencies:

-   PyTorch 1.9+
-   torchvision
-   timm (for additional backbone options)
-   LARS optimizer
-   Weights & Biases
-   tqdm

## 📊 Dataset Structure

The expected dataset structure:

```
dataset/
├── wearing/          # Images of people wearing fashion items
├── product/          # Product catalog images
└── train_metainfo.json  # Metadata linking wearing and product images
```

The metainfo JSON should contain entries with:

-   `wearing`: Path to wearing image
-   One or more product identifiers (e.g., `main_top`, `bottom`, etc.)

## 💻 Training

### Configuration

The training process is controlled through the `config.yaml` file. Key configurations:

```yaml
# Training settings
training:
    epochs: 50
    batch_size: 512 # Optimized for A100 80GB
    learning_rate: 1e-4
    num_workers: 8
    mixed_precision: true

# Model settings
model:
    backbone: "convnext_tiny" # Options: tiny, small, base, large
    embed_dim: 512
    image_size: 384

# Memory bank settings
memory_bank:
    enabled: true
    size: 16384 # Optimized for A100 80GB
    momentum: 0.99
    start_epoch: 1

# Monitoring settings
wandb:
    enabled: true
    project_name: "fashion-supcon-a100"
    log_gradients: true
    log_memory: true
    log_embedding_samples: 200
```

### Training Command

```bash
python train.py --config config.yaml
```

## 🔍 Key Components

### Memory Bank

The memory bank significantly extends the effective batch size by storing and utilizing recent embedding samples:

-   Maintains a queue of 16K+ normalized embeddings
-   Uses momentum updates to stabilize learning
-   Dramatically improves contrastive learning by providing more negative examples
-   Configurable start epoch and momentum rate

### WandB Monitoring

Comprehensive real-time monitoring is integrated with Weights & Biases:

-   **Training Metrics**: Loss, learning rate, top-k accuracy
-   **Validation Metrics**: Recall@K, precision, normalized mAP
-   **Visual Analysis**:
    -   Embedding distance visualizations
    -   Hard/easy sample identification
    -   Top-K retrieval examples
-   **System Monitoring**:
    -   GPU memory usage
    -   Gradient histograms
    -   Embedding space visualization

### Adaptive Hard Mining

The loss function incorporates adaptive temperature scaling to focus on hard negative examples:

-   Identifies and emphasizes hard negatives (similar but different products)
-   Applies temperature scaling based on difficulty
-   Improves separation in embedding space for challenging cases

## 📁 Directory Structure

```
.
├── README.md              # Project documentation
├── config.yaml            # Configuration file
├── train.py               # Main training script
├── models/
│   └── convnext.py        # Model architecture definitions
├── utils/
│   ├── config.py          # Configuration loader
│   ├── dataset.py         # Dataset and data loading utilities
│   ├── losses.py          # Contrastive loss implementations
│   ├── memory_bank.py     # Memory bank implementation
│   ├── metrics.py         # Evaluation metrics
│   ├── optimizers.py      # LARS optimizer
│   └── visualization.py   # Visualization utilities
├── inference/
│   ├── build_index.py     # Vector database construction
│   └── search.py          # Product retrieval functions
└── requirements.txt       # Python dependencies
```

## 📈 Evaluation and Inference

### Model Evaluation

During training, the model is evaluated on a validation set measuring:

-   Loss on validation pairs
-   Top-1/5/10 accuracy (retrieval performance)
-   Visualization of embedding distances

### Inference Pipeline

After training, the typical workflow is:

1. Generate product embeddings:

    ```bash
    python inference/build_index.py --model_path checkpoints/best_model.pth.tar --product_dir /path/to/products
    ```

2. Query with wearing images:
    ```bash
    python inference/search.py --query_image /path/to/query.jpg --top_k 5
    ```

## 🔗 References

-   [Supervised Contrastive Learning (SupCon)](https://arxiv.org/abs/2004.11362)
-   [ConvNeXt Architecture](https://arxiv.org/abs/2201.03545)
-   [LARS Optimizer](https://arxiv.org/abs/1708.03888)

## 📝 License

[MIT License](LICENSE)
