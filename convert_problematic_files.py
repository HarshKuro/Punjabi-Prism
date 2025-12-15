"""
Convert Problematic Audio Files to WAV Format
==============================================
Converts unplayable .m4a files to standard .wav format using ffmpeg or pydub
"""

import os
from pathlib import Path
from pydub import AudioSegment
import pandas as pd


def convert_m4a_to_wav(input_path, output_path=None, sample_rate=16000):
    """
    Convert .m4a file to .wav format.
    
    Args:
        input_path: Path to .m4a file
        output_path: Path for output .wav file (if None, replaces extension)
        sample_rate: Target sample rate in Hz
    
    Returns:
        bool: True if conversion successful
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.with_suffix('.wav')
    else:
        output_path = Path(output_path)
    
    try:
        print(f"Converting: {input_path.name}")
        
        # Load the .m4a file
        audio = AudioSegment.from_file(str(input_path), format="m4a")
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set sample rate
        audio = audio.set_frame_rate(sample_rate)
        
        # Export as WAV
        audio.export(
            str(output_path),
            format="wav",
            parameters=["-ar", str(sample_rate), "-ac", "1"]
        )
        
        print(f"  ✅ Saved to: {output_path.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Main conversion function."""
    print("🔧 Audio File Converter")
    print("=" * 80)
    
    # The two problematic files identified
    problematic_files = [
        r"C:\Users\Harsh Jain\Downloads\prism\Spoofed\f1\Spoofed-1\1m IP 14 Plus\pa_S01_f2_female_IP14Pl_MBP15_1m_90_east_60db_1_S.m4a",
        r"C:\Users\Harsh Jain\Downloads\prism\Spoofed\f1\Spoofed-1\1m IP 14 Plus\pa_S02_f2_female_IP14Pl_MBP15_1m_90_east_60db_1_S.m4a"
    ]
    
    print(f"\nFound {len(problematic_files)} files to convert:\n")
    
    success_count = 0
    fail_count = 0
    
    for file_path in problematic_files:
        if Path(file_path).exists():
            if convert_m4a_to_wav(file_path):
                success_count += 1
                # Optionally delete the original .m4a file
                # Path(file_path).unlink()
            else:
                fail_count += 1
        else:
            print(f"⚠️  File not found: {file_path}")
            fail_count += 1
    
    print("\n" + "=" * 80)
    print(f"✅ Successfully converted: {success_count} files")
    if fail_count > 0:
        print(f"❌ Failed to convert: {fail_count} files")
    print("=" * 80)
    
    # Load validation CSV to get full list of .m4a files
    print("\n🔍 Checking for other .m4a files in the dataset...")
    validation_csv = Path(r"C:\Users\Harsh Jain\Downloads\prism\validation_reports")
    
    # Find the latest validation CSV
    csv_files = list(validation_csv.glob("audio_validation_*.csv"))
    if csv_files:
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(latest_csv)
        
        # Find all .m4a files
        m4a_files = df[df['file_name'].str.endswith('.m4a', na=False)]
        
        if len(m4a_files) > 0:
            print(f"Found {len(m4a_files)} total .m4a files in dataset")
            print("\nWould you like to convert all .m4a files to .wav?")
            print("This ensures maximum compatibility with all systems.")
            
            # Save list of .m4a files for reference
            m4a_list_path = validation_csv / "m4a_files_list.csv"
            m4a_files[['relative_path', 'file_name', 'playable']].to_csv(m4a_list_path, index=False)
            print(f"\nSaved list of .m4a files to: {m4a_list_path}")
    
    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    main()
