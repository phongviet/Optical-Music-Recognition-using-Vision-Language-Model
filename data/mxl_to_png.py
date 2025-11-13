"""
Convert MusicXML (.mxl) files to PNG images using MuseScore 4.

This script recursively processes all .mxl files in the data/raw/mxl directory
and converts them to PNG format using MuseScore 4, storing the results in data/png
with the same directory structure.

Requirements:
    - MuseScore 4 must be installed
    - Install on Windows: Download from https://musescore.org/
    - Install on Ubuntu/Debian: sudo apt-get install musescore4
    - Install on macOS: brew install --cask musescore
"""

import subprocess
import sys
import os
from pathlib import Path
from tqdm import tqdm


def find_musescore():
    """
    Find the MuseScore 4 executable.

    Returns:
        str: Path to MuseScore 4 executable

    Raises:
        FileNotFoundError: If MuseScore 4 is not found
    """
    # Check if musescore4 is in PATH
    try:
        result = subprocess.run(['musescore4', '--version'],
                              capture_output=True,
                              timeout=5)
        if result.returncode == 0:
            return 'musescore4'
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Check common Windows locations
    if sys.platform == 'win32':
        common_paths = [
            r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe',
            r'C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe',
            os.path.expanduser(r'~\AppData\Local\Programs\MuseScore 4\bin\MuseScore4.exe'),
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path

    # Check common macOS location
    elif sys.platform == 'darwin':
        macos_path = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
        if os.path.exists(macos_path):
            return macos_path

    raise FileNotFoundError(
        "MuseScore 4 not found. Please install MuseScore 4:\n"
        "  Windows: Download from https://musescore.org/\n"
        "  Ubuntu/Debian: sudo apt-get install musescore4\n"
        "  macOS: brew install --cask musescore"
    )


def convert_mxl_to_png(mxl_path, png_path, musescore_cmd):
    """
    Convert a single MXL file to PNG format using MuseScore 4.

    Args:
        mxl_path (Path): Input MXL file path
        png_path (Path): Output PNG file path
        musescore_cmd (str): Path to MuseScore executable

    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        png_path.parent.mkdir(parents=True, exist_ok=True)

        # Run MuseScore conversion
        # MuseScore 4 uses -o flag for output
        result = subprocess.run(
            [musescore_cmd, '-o', str(png_path), str(mxl_path)],
            capture_output=True,
            text=True,
            timeout=60  # Longer timeout for rendering
        )

        # Check if output file was created
        if png_path.exists():
            return True
        else:
            # MuseScore might create file with -1 suffix for multi-page scores
            base_name = png_path.stem
            first_page = png_path.parent / f"{base_name}-1.png"
            if first_page.exists():
                return True

            print(f"Error converting {mxl_path.name}: Output file not created")
            if result.stderr:
                print(f"  Details: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"Timeout converting {mxl_path.name}")
        return False
    except Exception as e:
        print(f"Error converting {mxl_path.name}: {str(e)}")
        return False


def get_all_mxl_files(root_dir):
    """
    Recursively find all .mxl files in the directory.

    Args:
        root_dir (Path): Root directory to search

    Returns:
        list: List of Path objects for all .mxl files
    """
    mxl_files = []
    for path in root_dir.rglob('*.mxl'):
        mxl_files.append(path)
    return mxl_files


def main():
    """Main conversion process."""
    # Get the script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define input and output directories
    input_dir = project_root / 'data' / 'raw' / 'mxl'
    output_dir = project_root / 'data' / 'png'

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        sys.exit(1)

    # Find MuseScore executable
    try:
        musescore_cmd = find_musescore()
        print(f"Found MuseScore: {musescore_cmd}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Get all MXL files
    print("\nScanning for MXL files...")
    mxl_files = get_all_mxl_files(input_dir)
    print(f"Found {len(mxl_files)} MXL files")

    if len(mxl_files) == 0:
        print("No MXL files found!")
        sys.exit(0)

    # Convert each file
    print("\nConverting files to PNG...")
    successful = 0
    failed = 0

    for mxl_path in tqdm(mxl_files, desc="Converting"):
        # Calculate relative path from input_dir
        rel_path = mxl_path.relative_to(input_dir)

        # Create corresponding PNG path
        png_path = output_dir / rel_path.with_suffix('.png')

        # Skip if already converted (optional - comment out to reconvert)
        # Check for both single file and multi-page variants
        base_name = png_path.stem
        first_page = png_path.parent / f"{base_name}-1.png"
        if png_path.exists() or first_page.exists():
            successful += 1
            continue

        # Convert the file
        if convert_mxl_to_png(mxl_path, png_path, musescore_cmd):
            successful += 1
        else:
            failed += 1

    # Print summary
    print("\n" + "="*60)
    print(f"Conversion complete!")
    print(f"  Successful: {successful}/{len(mxl_files)}")
    print(f"  Failed: {failed}/{len(mxl_files)}")
    print(f"  Output directory: {output_dir}")
    print("\nNote: Multi-page scores may create multiple PNG files")
    print("      (e.g., name-1.png, name-2.png, etc.)")
    print("="*60)


if __name__ == '__main__':
    main()

