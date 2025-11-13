"""
Convert MusicXML (.mxl) files to ABC notation format using xml2abc.

This script recursively processes all .mxl files in the data/raw/mxl directory
and converts them to ABC format, storing the results in data/abc with the same
directory structure.

Requirements:
    - xml2abc tool must be installed (from abcMIDI package)
    - Install on Ubuntu/Debian: sudo apt-get install abcmidi
    - Install on macOS: brew install abcmidi
    - Install on Windows: Download from https://ifdo.ca/~seymour/runabc/top.html
"""

import subprocess
import sys
from pathlib import Path
from tqdm import tqdm


def find_xml2abc():
    """
    Find the xml2abc.py script.

    Returns:
        Path: Path to xml2abc.py script

    Raises:
        FileNotFoundError: If xml2abc.py is not found
    """
    # Get the script directory (data folder)
    script_dir = Path(__file__).parent
    xml2abc_path = script_dir / 'xml2abc.py'

    if xml2abc_path.exists():
        return xml2abc_path

    raise FileNotFoundError(
        f"xml2abc.py not found at {xml2abc_path}\n"
        "Please ensure xml2abc.py is in the data folder."
    )


def convert_mxl_to_abc(mxl_path, abc_path, xml2abc_script):
    """
    Convert a single MXL file to ABC format.

    Args:
        mxl_path (Path): Input MXL file path
        abc_path (Path): Output ABC file path
        xml2abc_script (Path): Path to xml2abc.py script

    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        abc_path.parent.mkdir(parents=True, exist_ok=True)

        # Run xml2abc conversion using Python
        # xml2abc.py writes to stdout by default, so we redirect it
        result = subprocess.run(
            [sys.executable, str(xml2abc_script), str(mxl_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            # Write the ABC content to file
            with open(abc_path, 'w', encoding='utf-8') as f:
                f.write(result.stdout)
            return True
        else:
            print(f"Error converting {mxl_path.name}: {result.stderr}")
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
    output_dir = project_root / 'data' / 'abc'

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        sys.exit(1)

    # Find xml2abc.py script
    try:
        xml2abc_script = find_xml2abc()
        print(f"Found xml2abc.py: {xml2abc_script}")
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
    print("\nConverting files...")
    successful = 0
    failed = 0

    for mxl_path in tqdm(mxl_files, desc="Converting"):
        # Calculate relative path from input_dir
        rel_path = mxl_path.relative_to(input_dir)

        # Create corresponding ABC path
        abc_path = output_dir / rel_path.with_suffix('.abc')

        # Skip if already converted (optional - comment out to reconvert)
        if abc_path.exists():
            successful += 1
            continue

        # Convert the file
        if convert_mxl_to_abc(mxl_path, abc_path, xml2abc_script):
            successful += 1
        else:
            failed += 1

    # Print summary
    print("\n" + "="*60)
    print(f"Conversion complete!")
    print(f"  Successful: {successful}/{len(mxl_files)}")
    print(f"  Failed: {failed}/{len(mxl_files)}")
    print(f"  Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

