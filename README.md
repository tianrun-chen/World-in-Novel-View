# World-in-Novel-View

[![Stars](https://img.shields.io/github/stars/tianrun-chen/World-in-Novel-View?style=social)](https://github.com/tianrun-chen/World-in-Novel-View/stargazers)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

**Scaling Novel View Synthesis for Static and Dynamic Scenes**
<div align="center">
[(im3.gif)](https://github.com/tianrun-chen/World-in-Novel-View/blob/main/im3.gif)
  
</div>

- **Distributed Training**: FSDP and DDP support for large-scale training
- **Mixed Precision**: Automatic mixed precision (AMP) for faster training
---
Train and Inference with Ascend 910b NPU
Torch.NPU is required。


## Project Structure
```
.
├── geope_core/              # Core GeoPE implementation
│   ├── torch.py             # PyTorch GeoPE attention
│   └── utils/               # Utility modules
│       ├── config.py        # Configuration utilities
│       ├── functional.py    # Functional utilities
│       ├── mha.py           # Multi-head attention
│       ├── runner.py        # Training runner
│       └── transformer.py   # Transformer components
├── src/                     # Application code
│   ├── geope_attention/     # GeoPE attention wrapper
│   ├── geope_utils/         # GeoPE utilities wrapper
│   ├── nvs_models/          # Novel view synthesis models
│   ├── nvs_data/            # Data loading and preprocessing
│   └── nvs_training/        # Training and evaluation
├── tests/                   # Unit tests
└── scripts/                 # Utility scripts
```
### Training

```bash
# Single GPU training
python -m src.main konet

# Multi-GPU training with FSDP
torchrun --standalone --nproc-per-node=4 -m src.main konet-fsdp
```



# Dataset: KOKONI-WorldVID-1A

KOKONI-WorldVID-1A is a large-scale video dataset designed for **Novel View Synthesis** research. It contains over **10,000 unique videos** sourced from **Bilibili**, one of China's leading video-sharing platforms.

## 💡 Dataset Highlights

Unlike most existing novel view synthesis datasets, KOKONI-WorldVID-1A provides videos from real-world, diverse scenarios with a unique data domain. These videos are created by a wide range of content creators, covering everything from static landscapes and object displays to dynamic human activities and lifestyle recordings.

- **Data Source**: All videos are sourced from Bilibili, providing the research community with a perspective distinct from Western-dominated datasets.
- **Scale**: Contains over 10,000 unique videos, offering sufficient data for deep learning model training.
- **Content Diversity**: Videos encompass a wide variety of content, helping to improve model generalization in complex real-world scenarios.
- **Static & Dynamic**: The dataset includes both static and partially dynamic videos. For static scene videos, we additionally provide human-screened static segments to facilitate more fine-grained model training and evaluation.

## 📊 Dataset Statistics

| Category | Count | Description |
|----------|-------|-------------|
| **Total Videos** | 10,000 | Unique videos from Bilibili |
| **Static Videos** | ~5,000 | Videos with static scenes and annotated segments |
| **Dynamic Videos** | ~5,000 | Videos with dynamic content (e.g., walking, movement) |
| **Data Domain** | Chinese UGC | User-generated content from China |
| **Application** | Novel View Synthesis | Training and evaluation of NVS models |

## 📂 Data Structure

The dataset is organized using two CSV files that contain metadata for all videos:

### 1. `static.csv` - Static Scene Videos

Contains videos with static scenes and human-annotated time segments.

**Format:**
```
序号,URL,视频标题,静态开始时间1,静态结束时间1,静态开始时间2,静态结束时间2
1,https://www.bilibili.com/video/BV1xx411c7mD,Beautiful Landscape,00:10,00:35,01:20,02:15
2,https://www.bilibili.com/video/BV1Ab411q7yH,Object Display,00:00,00:45,,
```

**Columns:**
- `序号` (Index): Sequential number
- `URL`: Bilibili video URL
- `视频标题` (Video Title): Original video title
- `静态开始时间1` (Static Start Time 1): Start time of first static segment (format: MM:SS or HH:MM:SS)
- `静态结束时间1` (Static End Time 1): End time of first static segment
- `静态开始时间2` (Static Start Time 2): Start time of second static segment (optional)
- `静态结束时间2` (Static End Time 2): End time of second static segment (optional)

**Note:** Additional segment pairs may exist (静态开始时间3, 静态结束时间3, etc.)

### 2. `walk.csv` - Dynamic Scene Videos

Contains videos with dynamic content such as walking, movement, or changing scenes.

**Format:**
```
序号,URL,视频标题
1,https://www.bilibili.com/video/BV1yZ4y1u7fA,City Walk Tour
2,https://www.bilibili.com/video/BV1Hx411v7iP,Campus Walking
```

**Columns:**
- `序号` (Index): Sequential number
- `URL`: Bilibili video URL
- `视频标题` (Video Title): Original video title

## 📥 Download & Usage

We provide a Python script (`download_videos.py`) to help users batch download videos from the dataset. Please follow these steps:

### 1. Install Dependencies

```bash
pip install you-get pandas
```

Or alternatively:

```bash
pip install yt-dlp pandas
```

### 2. Download All Videos

```bash
# Download all static videos
python download_videos.py --csv static.csv --output_dir ./videos/static

# Download all dynamic videos
python download_videos.py --csv walk.csv --output_dir ./videos/walk
```

### 3. Download Specific Range

```bash
# Download first 100 videos
python download_videos.py --csv static.csv --output_dir ./videos/static --start 0 --end 100
```

For detailed usage instructions, please refer to the [USAGE_GUIDE.md](USAGE_GUIDE.md).

## 🎯 Applications

KOKONI-WorldVID-1A is specifically designed for **Novel View Synthesis** research, including but not limited to:

- **3D Scene Reconstruction**: Reconstructing 3D scenes from video sequences
- **Neural Radiance Fields (NeRF)**: Training NeRF-based models on real-world data
- **3D Gaussian Splatting**: Learning 3D representations from video data
- **Multi-view Synthesis**: Generating novel viewpoints from limited observations
- **Dynamic Scene Modeling**: Handling both static and dynamic content

## ⚖️ License

The use of the KOKONI-WorldVID-1A dataset as a whole is licensed under the [ODC-By v1.0](https://opendatacommons.org/licenses/by/1.0/) license. 

**Important Notice:** All videos in this dataset are copyrighted by their original creators on Bilibili. This dataset is released solely for **non-commercial academic research purposes**. Any commercial use must obtain explicit permission from the original video creators. We are not responsible for any copyright disputes arising from the use of this dataset.
