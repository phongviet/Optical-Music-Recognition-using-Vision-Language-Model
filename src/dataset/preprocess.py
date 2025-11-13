"""
Preprocessing script to concatenate multi-page sheet music images and organize data.

This script:
1. Groups PNG files by their base name (without page numbers)
2. Concatenates all pages of the same sheet vertically
3. Saves concatenated images to data/processed/png/
4. Copies corresponding ABC files to data/processed/abc/
5. Splits data into train/test sets (80/20 by default)
6. Moves files to data/train/{png,abc}/ and data/test/{png,abc}/
7. Creates metadata CSV files:
   - train_metadata.csv: Training set mappings
   - test_metadata.csv: Test set mappings
   - metadata.csv: Combined mappings with 'split' column

Usage:
    # Command-line (with default 80/20 split):
    python src/dataset/preprocess.py --workers 8

    # Command-line (without split):
    python src/dataset/preprocess.py --workers 8 --no-split

    # Command-line (custom split ratio):
    python src/dataset/preprocess.py --workers 8 --train-ratio 0.9

    # Programmatic:
    from src.dataset.preprocess import preprocess_dataset
    preprocess_dataset(num_workers=8, train_test_split=True, train_ratio=0.8)
"""

import csv
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from PIL import Image
from tqdm import tqdm


def get_base_name(filename: str) -> str:
    """
    Extract base name from a PNG filename.

    Args:
        filename: PNG filename (e.g., 'Qma11MhFz64u8wRiW1cTZ5VJ4zTQWbYbVHDnMGafAFoaN6-1.png')

    Returns:
        Base name without page number (e.g., 'Qma11MhFz64u8wRiW1cTZ5VJ4zTQWbYbVHDnMGafAFoaN6')
    """
    # Remove .png extension
    name_without_ext = filename.replace('.png', '')
    # Remove page number suffix (e.g., '-1', '-2', etc.)
    if '-' in name_without_ext:
        parts = name_without_ext.rsplit('-', 1)
        if parts[-1].isdigit():
            return parts[0]
    return name_without_ext


def find_all_png_files(png_root: Path) -> Dict[str, List[Tuple[Path, int]]]:
    """
    Find all PNG files and group them by base name.

    Args:
        png_root: Root directory containing PNG files

    Returns:
        Dictionary mapping base_name to list of (file_path, page_number) tuples
    """
    grouped_files = defaultdict(list)

    # Walk through all PNG files
    for png_file in png_root.rglob('*.png'):
        filename = png_file.name
        base_name = get_base_name(filename)

        # Extract page number
        name_without_ext = filename.replace('.png', '')
        if '-' in name_without_ext:
            parts = name_without_ext.rsplit('-', 1)
            if parts[-1].isdigit():
                page_number = int(parts[-1])
            else:
                page_number = 1
        else:
            page_number = 1

        grouped_files[base_name].append((png_file, page_number))

    # Sort pages for each base name
    for base_name in grouped_files:
        grouped_files[base_name].sort(key=lambda x: x[1])

    return grouped_files


def find_abc_file(abc_root: Path, base_name: str) -> Optional[Path]:
    """
    Find the ABC file corresponding to a base name.

    Args:
        abc_root: Root directory containing ABC files
        base_name: Base name of the sheet music

    Returns:
        Path to ABC file, or None if not found
    """
    abc_filename = f"{base_name}.abc"

    # Search in all subdirectories
    for abc_file in abc_root.rglob(abc_filename):
        return abc_file

    return None


def concatenate_images_vertically(image_paths: List[Path]) -> Image.Image:
    """
    Concatenate multiple images vertically.

    Args:
        image_paths: List of paths to images to concatenate

    Returns:
        Concatenated PIL Image
    """
    images = [Image.open(img_path) for img_path in image_paths]

    # Get maximum width
    max_width = max(img.width for img in images)

    # Calculate total height
    total_height = sum(img.height for img in images)

    # Create new image with white background
    concatenated = Image.new('RGB', (max_width, total_height), (255, 255, 255))

    # Paste images
    current_y = 0
    for img in images:
        # Center image horizontally if it's narrower than max_width
        x_offset = (max_width - img.width) // 2
        concatenated.paste(img, (x_offset, current_y))
        current_y += img.height

    # Close opened images
    for img in images:
        img.close()

    return concatenated


def process_single_sheet(
    base_name: str,
    page_list: List[Tuple[Path, int]],
    abc_root: Path,
    output_png_dir: Path,
    output_abc_dir: Path
) -> Optional[Dict[str, str]]:
    """
    Process a single sheet music: concatenate pages and copy ABC file.

    Args:
        base_name: Base name of the sheet music
        page_list: List of (file_path, page_number) tuples
        abc_root: Root directory containing ABC files
        output_png_dir: Output directory for processed PNG files
        output_abc_dir: Output directory for processed ABC files

    Returns:
        Metadata dictionary with png_path and abc_path, or None if processing failed
    """
    try:
        # Find corresponding ABC file
        abc_file = find_abc_file(abc_root, base_name)

        if abc_file is None:
            return None

        # Extract just the file paths from (path, page_number) tuples
        page_paths = [path for path, _ in page_list]

        # Concatenate images
        concatenated_img = concatenate_images_vertically(page_paths)

        # Save concatenated image
        output_png_path = output_png_dir / f"{base_name}.png"
        concatenated_img.save(output_png_path)
        concatenated_img.close()

        # Copy ABC file
        output_abc_path = output_abc_dir / f"{base_name}.abc"
        shutil.copy2(abc_file, output_abc_path)

        # Create metadata entry
        # Store relative paths from data/ directory
        relative_png_path = output_png_path.relative_to(Path("data"))
        relative_abc_path = output_abc_path.relative_to(Path("data"))

        return {
            'png_path': str(relative_png_path).replace('\\', '/'),
            'abc_path': str(relative_abc_path).replace('\\', '/')
        }

    except Exception as e:
        print(f"\nError processing {base_name}: {e}")
        return None


def split_train_test(
    metadata: List[Dict[str, str]],
    processed_png_dir: Path,
    processed_abc_dir: Path,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split processed files into train and test sets.

    Args:
        metadata: List of metadata dictionaries with png_path and abc_path
        processed_png_dir: Directory containing processed PNG files
        processed_abc_dir: Directory containing processed ABC files
        train_ratio: Ratio of training data (default: 0.8 for 80/20 split)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_metadata, test_metadata)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Shuffle metadata
    shuffled_metadata = metadata.copy()
    random.shuffle(shuffled_metadata)

    # Calculate split index
    total_count = len(shuffled_metadata)
    train_count = int(total_count * train_ratio)

    # Split metadata
    train_metadata = shuffled_metadata[:train_count]
    test_metadata = shuffled_metadata[train_count:]

    # Create train/test directories
    train_png_dir = processed_png_dir.parent / "train" / "png"
    train_abc_dir = processed_png_dir.parent / "train" / "abc"
    test_png_dir = processed_png_dir.parent / "test" / "png"
    test_abc_dir = processed_png_dir.parent / "test" / "abc"

    train_png_dir.mkdir(parents=True, exist_ok=True)
    train_abc_dir.mkdir(parents=True, exist_ok=True)
    test_png_dir.mkdir(parents=True, exist_ok=True)
    test_abc_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSplitting dataset: {train_count} train / {len(test_metadata)} test")
    print("Moving files to train/test directories...")

    # Move train files
    train_metadata_updated = []
    for entry in tqdm(train_metadata, desc="Moving train files"):
        # Get original paths
        png_path = Path("data") / entry['png_path']
        abc_path = Path("data") / entry['abc_path']

        # Get filenames
        png_filename = png_path.name
        abc_filename = abc_path.name

        # Move to train directory
        new_png_path = train_png_dir / png_filename
        new_abc_path = train_abc_dir / abc_filename

        shutil.move(str(png_path), str(new_png_path))
        shutil.move(str(abc_path), str(new_abc_path))

        # Update metadata with new paths
        train_metadata_updated.append({
            'png_path': str(new_png_path.relative_to(Path("data"))).replace('\\', '/'),
            'abc_path': str(new_abc_path.relative_to(Path("data"))).replace('\\', '/')
        })

    # Move test files
    test_metadata_updated = []
    for entry in tqdm(test_metadata, desc="Moving test files"):
        # Get original paths
        png_path = Path("data") / entry['png_path']
        abc_path = Path("data") / entry['abc_path']

        # Get filenames
        png_filename = png_path.name
        abc_filename = abc_path.name

        # Move to test directory
        new_png_path = test_png_dir / png_filename
        new_abc_path = test_abc_dir / abc_filename

        shutil.move(str(png_path), str(new_png_path))
        shutil.move(str(abc_path), str(new_abc_path))

        # Update metadata with new paths
        test_metadata_updated.append({
            'png_path': str(new_png_path.relative_to(Path("data"))).replace('\\', '/'),
            'abc_path': str(new_abc_path.relative_to(Path("data"))).replace('\\', '/')
        })

    return train_metadata_updated, test_metadata_updated


def preprocess_dataset(
    png_root: str = "data/png",
    abc_root: str = "data/abc",
    output_png_dir: str = "data/processed/png",
    output_abc_dir: str = "data/processed/abc",
    metadata_csv: str = "data/metadata.csv",
    num_workers: int = None,
    train_test_split: bool = True,
    train_ratio: float = 0.8,
    random_seed: int = 42
):
    """
    Main preprocessing function with multithreading support.

    Args:
        png_root: Root directory containing source PNG files
        abc_root: Root directory containing source ABC files
        output_png_dir: Output directory for processed PNG files
        output_abc_dir: Output directory for processed ABC files
        metadata_csv: Output CSV file path
        num_workers: Number of worker threads (default: None = number of CPU cores)
        train_test_split: Whether to split data into train/test sets (default: True)
        train_ratio: Ratio of training data (default: 0.8 for 80/20 split)
        random_seed: Random seed for reproducibility (default: 42)
    """
    # Convert to Path objects
    png_root = Path(png_root)
    abc_root = Path(abc_root)
    output_png_dir = Path(output_png_dir)
    output_abc_dir = Path(output_abc_dir)
    metadata_csv = Path(metadata_csv)

    # Create output directories
    output_png_dir.mkdir(parents=True, exist_ok=True)
    output_abc_dir.mkdir(parents=True, exist_ok=True)

    print("Finding all PNG files...")
    grouped_pngs = find_all_png_files(png_root)
    print(f"Found {len(grouped_pngs)} unique sheet music pieces")

    # List to store metadata entries (thread-safe)
    metadata = []
    metadata_lock = Lock()

    # Counters (thread-safe)
    processed_count = 0
    skipped_count = 0
    counter_lock = Lock()

    # Process each sheet music with multithreading
    print(f"\nProcessing sheet music files with {num_workers or 'auto'} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_name = {
            executor.submit(
                process_single_sheet,
                base_name,
                page_list,
                abc_root,
                output_png_dir,
                output_abc_dir
            ): base_name
            for base_name, page_list in grouped_pngs.items()
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(future_to_name), desc="Processing") as pbar:
            for future in as_completed(future_to_name):
                base_name = future_to_name[future]

                try:
                    result = future.result()

                    if result is not None:
                        # Add to metadata (thread-safe)
                        with metadata_lock:
                            metadata.append(result)

                        with counter_lock:
                            processed_count += 1
                    else:
                        with counter_lock:
                            skipped_count += 1

                except Exception as e:
                    print(f"\nUnexpected error processing {base_name}: {e}")
                    with counter_lock:
                        skipped_count += 1

                pbar.update(1)

    print(f"\nProcessed: {processed_count} sheet music pieces")
    print(f"Skipped: {skipped_count} pieces (no matching ABC file)")

    # Split into train/test if requested
    if train_test_split and len(metadata) > 0:
        print(f"\n{'='*60}")
        print("Train/Test Split")
        print(f"{'='*60}")

        train_metadata, test_metadata = split_train_test(
            metadata,
            output_png_dir,
            output_abc_dir,
            train_ratio,
            random_seed
        )

        # Write separate metadata files for train and test
        train_csv = metadata_csv.parent / "train_metadata.csv"
        test_csv = metadata_csv.parent / "test_metadata.csv"

        print(f"\nWriting train metadata to {train_csv}...")
        with open(train_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['png_path', 'abc_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(train_metadata)

        print(f"Writing test metadata to {test_csv}...")
        with open(test_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['png_path', 'abc_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(test_metadata)

        # Write combined metadata
        print(f"Writing combined metadata to {metadata_csv}...")
        combined_metadata = train_metadata + test_metadata
        with open(metadata_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['png_path', 'abc_path', 'split']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for entry in train_metadata:
                entry_with_split = entry.copy()
                entry_with_split['split'] = 'train'
                writer.writerow(entry_with_split)

            for entry in test_metadata:
                entry_with_split = entry.copy()
                entry_with_split['split'] = 'test'
                writer.writerow(entry_with_split)

        print(f"\n{'='*60}")
        print(f"Preprocessing complete!")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_metadata)}")
        print(f"Test samples: {len(test_metadata)}")
        print(f"Train ratio: {train_ratio:.1%}")
        print(f"Train metadata: {train_csv}")
        print(f"Test metadata: {test_csv}")
        print(f"Combined metadata: {metadata_csv}")
        print(f"{'='*60}")
    else:
        # Write metadata CSV without split
        print(f"\nWriting metadata to {metadata_csv}...")
        with open(metadata_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['png_path', 'abc_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(metadata)

        print(f"\n{'='*60}")
        print(f"Preprocessing complete!")
        print(f"{'='*60}")
        print(f"Processed: {processed_count} sheet music pieces")
        print(f"Skipped: {skipped_count} pieces (no matching ABC file)")
        print(f"Output PNG directory: {output_png_dir}")
        print(f"Output ABC directory: {output_abc_dir}")
        print(f"Metadata CSV: {metadata_csv}")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess sheet music dataset: concatenate multi-page images and create metadata'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker threads (default: auto-detect based on CPU cores)'
    )

    parser.add_argument(
        '--png-root',
        type=str,
        default='data/png',
        help='Root directory containing source PNG files (default: data/png)'
    )

    parser.add_argument(
        '--abc-root',
        type=str,
        default='data/abc',
        help='Root directory containing source ABC files (default: data/abc)'
    )

    parser.add_argument(
        '--output-png',
        type=str,
        default='data/processed/png',
        help='Output directory for processed PNG files (default: data/processed/png)'
    )

    parser.add_argument(
        '--output-abc',
        type=str,
        default='data/processed/abc',
        help='Output directory for processed ABC files (default: data/processed/abc)'
    )

    parser.add_argument(
        '--metadata',
        type=str,
        default='data/metadata.csv',
        help='Output CSV file path (default: data/metadata.csv)'
    )

    parser.add_argument(
        '--no-split',
        action='store_true',
        help='Disable train/test split (default: split is enabled)'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Ratio of training data (default: 0.8 for 80/20 split)'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for train/test split (default: 42)'
    )

    args = parser.parse_args()

    print("="*60)
    print("Sheet Music Preprocessing Pipeline")
    print("="*60)
    print(f"PNG source: {args.png_root}")
    print(f"ABC source: {args.abc_root}")
    print(f"PNG output: {args.output_png}")
    print(f"ABC output: {args.output_abc}")
    print(f"Metadata CSV: {args.metadata}")
    print(f"Workers: {args.workers or 'auto'}")
    print(f"Train/Test Split: {'No' if args.no_split else 'Yes'}")
    if not args.no_split:
        print(f"Train Ratio: {args.train_ratio:.1%}")
        print(f"Random Seed: {args.random_seed}")
    print("="*60)
    print()

    # Run preprocessing
    preprocess_dataset(
        png_root=args.png_root,
        abc_root=args.abc_root,
        output_png_dir=args.output_png,
        output_abc_dir=args.output_abc,
        metadata_csv=args.metadata,
        num_workers=args.workers,
        train_test_split=not args.no_split,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed
    )

