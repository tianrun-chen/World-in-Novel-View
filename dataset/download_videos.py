#!/usr/bin/env python3
"""
KOKONI-WorldVID-1A Video Downloader

This script downloads videos from Bilibili based on CSV files (static.csv and walk.csv).
It supports batch downloading and provides progress tracking.

Requirements:
    - you-get: pip install you-get
    - Or yt-dlp: pip install yt-dlp (alternative)
    - pandas: pip install pandas

Usage:
    python download_videos.py --csv static.csv --output_dir ./videos/static
    python download_videos.py --csv walk.csv --output_dir ./videos/walk
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd
except ImportError:
    print("✗ Error: pandas is not installed")
    print("  Please install it with: pip install pandas")
    sys.exit(1)


def extract_video_id(url: str) -> str:
    """Extract video ID (BV number) from Bilibili URL.
    
    Args:
        url: Bilibili video URL
        
    Returns:
        Video ID (BV number)
    """
    # Match BV number from URL
    match = re.search(r'(BV[a-zA-Z0-9]+)', url)
    if match:
        return match.group(1)
    
    # If no BV found, create a simple ID from URL
    return url.split('/')[-1].split('?')[0]


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load video metadata from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame containing video metadata
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"✓ Loaded {len(df)} videos from CSV file")
        return df
    except FileNotFoundError:
        print(f"✗ Error: CSV file '{csv_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error reading CSV file: {e}")
        sys.exit(1)


def parse_static_segments(row: pd.Series) -> List[Tuple[str, str]]:
    """Parse static time segments from CSV row.
    
    Args:
        row: DataFrame row containing segment information
        
    Returns:
        List of (start_time, end_time) tuples
    """
    segments = []
    
    # Check for segment columns (静态开始时间1, 静态结束时间1, etc.)
    i = 1
    while True:
        start_col = f'静态开始时间{i}'
        end_col = f'静态结束时间{i}'
        
        if start_col not in row or end_col not in row:
            break
        
        start_time = row[start_col]
        end_time = row[end_col]
        
        # Check if both times are valid (not NaN)
        if pd.notna(start_time) and pd.notna(end_time):
            segments.append((str(start_time), str(end_time)))
        
        i += 1
    
    return segments


def check_dependencies():
    """Check if required dependencies are installed."""
    # Check for you-get
    try:
        subprocess.run(['you-get', '--version'], 
                      capture_output=True, 
                      check=True)
        print("✓ you-get is installed")
        return 'you-get'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check for yt-dlp as alternative
    try:
        subprocess.run(['yt-dlp', '--version'], 
                      capture_output=True, 
                      check=True)
        print("✓ yt-dlp is installed")
        return 'yt-dlp'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    print("✗ Error: Neither you-get nor yt-dlp is installed")
    print("  Please install one of them:")
    print("    pip install you-get")
    print("    pip install yt-dlp")
    sys.exit(1)


def download_video_youget(url: str, output_dir: str, video_id: str) -> bool:
    """Download a video using you-get.
    
    Args:
        url: Video URL
        output_dir: Output directory
        video_id: Video ID for naming
        
    Returns:
        True if download succeeded, False otherwise
    """
    try:
        cmd = [
            'you-get',
            '--output-dir', output_dir,
            '--output-filename', video_id,
            url
        ]
        
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True,
                              timeout=600)  # 10 minutes timeout
        
        if result.returncode == 0:
            return True
        else:
            print(f"  Error output: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  Timeout: Download took too long")
        return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def download_video_ytdlp(url: str, output_dir: str, video_id: str) -> bool:
    """Download a video using yt-dlp.
    
    Args:
        url: Video URL
        output_dir: Output directory
        video_id: Video ID for naming
        
    Returns:
        True if download succeeded, False otherwise
    """
    try:
        output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")
        
        cmd = [
            'yt-dlp',
            '--output', output_template,
            '--no-playlist',
            url
        ]
        
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True,
                              timeout=600)  # 10 minutes timeout
        
        if result.returncode == 0:
            return True
        else:
            print(f"  Error output: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  Timeout: Download took too long")
        return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def save_metadata_json(df: pd.DataFrame, output_path: str, csv_type: str) -> None:
    """Save metadata to JSON format for easier processing.
    
    Args:
        df: DataFrame containing metadata
        output_path: Path to save JSON file
        csv_type: Type of CSV ('static' or 'walk')
    """
    import json
    
    metadata = []
    
    for idx, row in df.iterrows():
        url = row['URL']
        video_id = extract_video_id(url)
        title = row['视频标题']
        
        entry = {
            'index': int(row['序号']) if '序号' in row else idx + 1,
            'video_id': video_id,
            'url': url,
            'title': title,
            'type': csv_type
        }
        
        # Add static segments if this is from static.csv
        if csv_type == 'static':
            segments = parse_static_segments(row)
            entry['static_segments'] = [
                {'start_time': start, 'end_time': end}
                for start, end in segments
            ]
        
        metadata.append(entry)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved metadata to {output_path}")


def download_videos(df: pd.DataFrame, 
                   output_dir: str, 
                   downloader: str,
                   start_idx: int = 0,
                   end_idx: int = None,
                   csv_type: str = 'unknown') -> None:
    """Download all videos from DataFrame.
    
    Args:
        df: DataFrame containing video metadata
        output_dir: Output directory for videos
        downloader: Downloader to use ('you-get' or 'yt-dlp')
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive), None means download all
        csv_type: Type of CSV ('static' or 'walk')
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine range
    if end_idx is None:
        end_idx = len(df)
    
    # Ensure valid range
    start_idx = max(0, start_idx)
    end_idx = min(len(df), end_idx)
    
    total = end_idx - start_idx
    successful = 0
    failed = 0
    skipped = 0
    
    print(f"\nStarting download of {total} videos ({csv_type})...")
    print(f"Output directory: {output_dir}\n")
    
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        url = row['URL']
        title = row['视频标题']
        video_id = extract_video_id(url)
        
        print(f"[{idx + 1}/{len(df)}] Downloading: {title}")
        print(f"  Video ID: {video_id}")
        print(f"  URL: {url}")
        
        # Show static segments if available
        if csv_type == 'static':
            segments = parse_static_segments(row)
            if segments:
                print(f"  Static Segments: {len(segments)}")
                for i, (start, end) in enumerate(segments, 1):
                    print(f"    Segment {i}: {start} - {end}")
        
        # Check if already downloaded
        existing_files = list(Path(output_dir).glob(f"{video_id}.*"))
        if existing_files:
            print(f"  ⊙ Already exists: {existing_files[0].name}")
            skipped += 1
            print()
            continue
        
        # Download
        if downloader == 'you-get':
            success = download_video_youget(url, output_dir, video_id)
        else:  # yt-dlp
            success = download_video_ytdlp(url, output_dir, video_id)
        
        if success:
            print(f"  ✓ Downloaded successfully")
            successful += 1
        else:
            print(f"  ✗ Download failed")
            failed += 1
        
        print()
    
    # Summary
    print("=" * 60)
    print(f"Download Summary:")
    print(f"  Total: {total}")
    print(f"  Successful: {successful}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Download videos from Bilibili for KOKONI-WorldVID-1A dataset'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file (static.csv or walk.csv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./videos',
        help='Output directory for downloaded videos (default: ./videos)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Starting index (inclusive, default: 0)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='Ending index (exclusive, default: all)'
    )
    parser.add_argument(
        '--save_json',
        type=str,
        default=None,
        help='Save metadata to JSON file (optional)'
    )
    
    args = parser.parse_args()
    
    # Determine CSV type
    csv_filename = Path(args.csv).name.lower()
    if 'static' in csv_filename:
        csv_type = 'static'
    elif 'walk' in csv_filename:
        csv_type = 'walk'
    else:
        csv_type = 'unknown'
    
    print(f"CSV Type: {csv_type}")
    
    # Check dependencies
    downloader = check_dependencies()
    
    # Load CSV
    df = load_csv(args.csv)
    
    # Save to JSON if requested
    if args.save_json:
        save_metadata_json(df, args.save_json, csv_type)
    
    # Download videos
    download_videos(df, args.output_dir, downloader, args.start, args.end, csv_type)


if __name__ == '__main__':
    main()

