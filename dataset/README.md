# KOKONI-WorldVID-1A Usage Guide

This guide will help you quickly get started with the KOKONI-WorldVID-1A dataset and batch download script.

## 📋 Prerequisites

Before using the download script, you need to install the following dependencies:

### Method 1: Using you-get (Recommended)

```bash
pip install you-get pandas
```

### Method 2: Using yt-dlp (Alternative)

```bash
pip install yt-dlp pandas
```

The script will automatically detect which download tool you have installed and use it.

## 🚀 Quick Start

### 1. Prepare CSV Files

The dataset uses two CSV files to organize video metadata:

#### `static.csv` - Static Scene Videos

Contains videos with static scenes and human-annotated time segments.

**Format:**
```csv
序号,URL,视频标题,静态开始时间1,静态结束时间1,静态开始时间2,静态结束时间2
1,https://www.bilibili.com/video/BV1xx411c7mD,Beautiful Landscape,00:10,00:35,01:20,02:15
2,https://www.bilibili.com/video/BV1Ab411q7yH,Object Display,00:00,00:45,,
```

**Columns:**
- `序号` (Index): Sequential number
- `URL`: Bilibili video URL
- `视频标题` (Video Title): Original video title
- `静态开始时间1` (Static Start Time 1): Start time of first static segment
- `静态结束时间1` (Static End Time 1): End time of first static segment
- `静态开始时间2` (Static Start Time 2): Start time of second static segment (optional)
- `静态结束时间2` (Static End Time 2): End time of second static segment (optional)

#### `walk.csv` - Dynamic Scene Videos

Contains videos with dynamic content.

**Format:**
```csv
序号,URL,视频标题
1,https://www.bilibili.com/video/BV1yZ4y1u7fA,City Walk Tour
2,https://www.bilibili.com/video/BV1Hx411v7iP,Campus Walking
```

**Columns:**
- `序号` (Index): Sequential number
- `URL`: Bilibili video URL
- `视频标题` (Video Title): Original video title

We provide example files `static_example.csv` and `walk_example.csv` for your reference.

### 2. Download All Videos

Run the following commands to download all videos from the CSV files:

```bash
# Download all static videos
python download_videos.py --csv static.csv --output_dir ./videos/static

# Download all dynamic videos
python download_videos.py --csv walk.csv --output_dir ./videos/walk
```

**Parameters:**
- `--csv`: Path to the CSV file (required)
- `--output_dir`: Output directory for videos (default: `./videos`)

### 3. Download Partial Videos

If you only want to download a subset of videos, use the `--start` and `--end` parameters:

```bash
# Download first 100 videos (indices 0-99)
python download_videos.py --csv static.csv --output_dir ./videos/static --start 0 --end 100

# Download videos 101-200 (indices 100-199)
python download_videos.py --csv static.csv --output_dir ./videos/static --start 100 --end 200
```

### 4. Export Metadata to JSON

You can export the CSV metadata to JSON format for easier processing:

```bash
python download_videos.py --csv static.csv --save_json static_metadata.json
```

This creates a JSON file with structured metadata including video IDs, URLs, titles, and static segments.

## 📊 Script Features

### ✓ Automatic Video ID Extraction

The script automatically extracts Bilibili video IDs (BV numbers) from URLs for consistent file naming.

### ✓ Skip Already Downloaded Videos

The script automatically detects existing video files in the output directory and skips re-downloading them.

### ✓ Progress Tracking

The script displays detailed progress information:
- Current video number and total count
- Video title and ID
- URL being downloaded
- Static segments (for static videos)
- Download status (success/failed/already exists)

### ✓ Download Summary

After completion, the script displays a summary:
- Total videos processed
- Successfully downloaded
- Skipped (already exists)
- Failed downloads

### ✓ Timeout Protection

Each video download has a 10-minute timeout to prevent hanging on problematic videos.

### ✓ Static Segment Information

For static videos, the script displays all annotated static segments during download, making it easy to track which videos have multiple segments.

## 🔧 Advanced Usage

### Batch Download Large-Scale Dataset

For the complete dataset with 10,000 videos, we recommend downloading in batches:

```bash
# Batch 1: 0-1000
python download_videos.py --csv static.csv --output_dir ./videos/static --start 0 --end 1000

# Batch 2: 1000-2000
python download_videos.py --csv static.csv --output_dir ./videos/static --start 1000 --end 2000

# Batch 3: 2000-3000
python download_videos.py --csv static.csv --output_dir ./videos/static --start 2000 --end 3000

# Continue for remaining batches...
```

### Extract Static Segments

For videos marked as static with annotated segments, you can use FFmpeg to extract these segments:

```bash
# Extract a single segment
ffmpeg -i BV1xx411c7mD.mp4 -ss 00:10 -to 00:35 -c copy BV1xx411c7mD_segment_1.mp4

# Extract multiple segments from the same video
ffmpeg -i BV1xx411c7mD.mp4 -ss 00:10 -to 00:35 -c copy BV1xx411c7mD_segment_1.mp4
ffmpeg -i BV1xx411c7mD.mp4 -ss 01:20 -to 02:15 -c copy BV1xx411c7mD_segment_2.mp4
```

### Automated Segment Extraction Script

Here's a Python script to automatically extract all static segments:

```python
import pandas as pd
import subprocess
from pathlib import Path

def extract_segments(csv_path, video_dir, output_dir):
    df = pd.read_csv(csv_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for idx, row in df.iterrows():
        video_id = extract_video_id(row['URL'])
        video_file = list(Path(video_dir).glob(f"{video_id}.*"))[0]
        
        # Extract each segment
        i = 1
        while True:
            start_col = f'静态开始时间{i}'
            end_col = f'静态结束时间{i}'
            
            if start_col not in row or pd.isna(row[start_col]):
                break
            
            start_time = row[start_col]
            end_time = row[end_col]
            output_file = Path(output_dir) / f"{video_id}_segment_{i}.mp4"
            
            cmd = [
                'ffmpeg', '-i', str(video_file),
                '-ss', start_time, '-to', end_time,
                '-c', 'copy', str(output_file)
            ]
            
            subprocess.run(cmd, capture_output=True)
            print(f"Extracted: {output_file.name}")
            i += 1

# Usage
extract_segments('static.csv', './videos/static', './segments')
```

### Organize Videos by Type

```bash
# Create organized directory structure
mkdir -p dataset/static
mkdir -p dataset/dynamic

# Download to organized directories
python download_videos.py --csv static.csv --output_dir dataset/static
python download_videos.py --csv walk.csv --output_dir dataset/dynamic
```

## ⚠️ Important Notes

1. **Network Connection**: Ensure you have a stable internet connection. Downloading large numbers of videos may take considerable time.

2. **Storage Space**: 10,000 videos may require several hundred GB of storage. Ensure you have sufficient disk space.

3. **Copyright**: All videos are copyrighted by their original creators. Use this dataset only for academic research purposes.

4. **Rate Limiting**: Bilibili may impose rate limits on frequent download requests. Consider adding delays between downloads if needed.

5. **Video Formats**: Downloaded video formats depend on what Bilibili provides, typically MP4 or FLV.

6. **CSV Encoding**: The CSV files use UTF-8 encoding with Chinese headers. Ensure your system supports UTF-8.

## 🐛 Troubleshooting

### Issue: Download Fails

**Solutions:**
- Check your network connection
- Verify the video URL is still valid
- Try manually accessing the video link to confirm it hasn't been deleted or made private
- Update you-get or yt-dlp to the latest version:
  ```bash
  pip install --upgrade you-get
  # or
  pip install --upgrade yt-dlp
  ```

### Issue: Dependency Installation Fails

**Solutions:**
```bash
# Use a mirror (for users in China)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple you-get pandas

# Or use conda
conda install -c conda-forge you-get pandas
```

### Issue: Permission Error

**Solutions:**
```bash
# Ensure the script has execute permission
chmod +x download_videos.py

# Or run with python explicitly
python download_videos.py --csv static.csv --output_dir ./videos
```

### Issue: CSV Encoding Error

**Solutions:**
- Ensure your CSV files are saved with UTF-8 encoding
- If you're editing CSV files in Excel, use "CSV UTF-8" format when saving
- On Windows, you may need to specify encoding explicitly in the script

### Issue: FFmpeg Not Found (for segment extraction)

**Solutions:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## 📞 Getting Help

To view all available parameters:

```bash
python download_videos.py --help
```

**Output:**
```
usage: download_videos.py [-h] --csv CSV [--output_dir OUTPUT_DIR] 
                          [--start START] [--end END] [--save_json SAVE_JSON]

Download videos from Bilibili for KOKONI-WorldVID-1A dataset

optional arguments:
  -h, --help            show this help message and exit
  --csv CSV             Path to CSV file (static.csv or walk.csv)
  --output_dir OUTPUT_DIR
                        Output directory for downloaded videos (default: ./videos)
  --start START         Starting index (inclusive, default: 0)
  --end END             Ending index (exclusive, default: all)
  --save_json SAVE_JSON
                        Save metadata to JSON file (optional)
```

## 🎯 Best Practices

1. **Test with Small Batches**: Before downloading the entire dataset, test with a small batch (e.g., 10-20 videos) to ensure everything works correctly.

2. **Regular Backups**: Periodically backup downloaded videos to prevent accidental loss.

3. **Use Logging**: Redirect output to a log file for later review:
   ```bash
   python download_videos.py --csv static.csv --output_dir ./videos 2>&1 | tee download.log
   ```

4. **Parallel Downloads**: For faster downloading, split the dataset and download in parallel on different machines or processes:
   ```bash
   # Terminal 1
   python download_videos.py --csv static.csv --output_dir ./videos --start 0 --end 2500
   
   # Terminal 2
   python download_videos.py --csv static.csv --output_dir ./videos --start 2500 --end 5000
   
   # Terminal 3
   python download_videos.py --csv static.csv --output_dir ./videos --start 5000 --end 7500
   
   # Terminal 4
   python download_videos.py --csv static.csv --output_dir ./videos --start 7500 --end 10000
   ```

5. **Monitor Disk Space**: Regularly check available disk space during large downloads:
   ```bash
   df -h
   ```

6. **Verify Downloads**: After downloading, verify file integrity by checking file sizes and attempting to play videos.

7. **Organize Metadata**: Keep the original CSV files and any generated JSON metadata files together with the videos for future reference.

## 📈 Dataset Statistics

After downloading, you can generate statistics about your dataset:

```python
import pandas as pd
from pathlib import Path

# Load CSV
df = pd.read_csv('static.csv')

# Count videos with multiple segments
multi_segment = 0
for idx, row in df.iterrows():
    if pd.notna(row.get('静态开始时间2', None)):
        multi_segment += 1

print(f"Total static videos: {len(df)}")
print(f"Videos with multiple segments: {multi_segment}")

# Check downloaded files
video_dir = Path('./videos/static')
downloaded = len(list(video_dir.glob('*.mp4'))) + len(list(video_dir.glob('*.flv')))
print(f"Downloaded videos: {downloaded}")
print(f"Download completion: {downloaded/len(df)*100:.1f}%")
```

## 🔗 Useful Resources

- **Bilibili**: https://www.bilibili.com
- **you-get Documentation**: https://github.com/soimort/you-get
- **yt-dlp Documentation**: https://github.com/yt-dlp/yt-dlp
- **FFmpeg Documentation**: https://ffmpeg.org/documentation.html
- **Pandas Documentation**: https://pandas.pydata.org/docs/

---

Happy downloading! If you encounter any issues, please open an issue on our GitHub repository.

