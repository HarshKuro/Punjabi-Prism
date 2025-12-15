"""
Audio File Validation and Naming Convention Script
==================================================
Purpose: Check if audio files are playable on Windows and apply conventional naming

Features:
- Scans all audio files in Bonafide and Spoofed directories
- Tests each file for playability using librosa
- Identifies corrupted or incompatible files
- Applies conventional naming (removes .m4a.wav double extensions)
- Generates detailed report of issues
"""

import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AudioFileValidator:
    """Validates audio files and checks for playability issues."""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.results = []
        self.summary = {
            'total_files': 0,
            'playable': 0,
            'unplayable': 0,
            'renamed': 0,
            'errors': []
        }
    
    def check_audio_file(self, file_path):
        """
        Check if an audio file is playable on Windows.
        
        Returns:
            dict: Status information about the file
        """
        file_info = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size_kb': file_path.stat().st_size / 1024,
            'playable': False,
            'error': None,
            'duration': None,
            'sample_rate': None,
            'channels': None,
            'needs_rename': False,
            'suggested_name': None
        }
        
        try:
            # Try to load the audio file
            y, sr = librosa.load(str(file_path), sr=None, mono=False)
            
            # Check if audio data is valid
            if y is None or len(y) == 0:
                file_info['error'] = "Empty audio data"
                file_info['playable'] = False
            elif np.all(y == 0):
                file_info['error'] = "Audio contains only silence/zeros"
                file_info['playable'] = False
            elif np.any(np.isnan(y)) or np.any(np.isinf(y)):
                file_info['error'] = "Audio contains NaN or Inf values"
                file_info['playable'] = False
            else:
                # File is playable
                file_info['playable'] = True
                file_info['sample_rate'] = sr
                file_info['duration'] = librosa.get_duration(y=y, sr=sr)
                file_info['channels'] = 1 if y.ndim == 1 else y.shape[0]
                
        except Exception as e:
            file_info['error'] = str(e)
            file_info['playable'] = False
        
        # Check for naming issues (double extensions like .m4a.wav)
        if '.m4a.wav' in file_path.name.lower():
            file_info['needs_rename'] = True
            file_info['suggested_name'] = file_path.name.replace('.m4a.wav', '.wav').replace('.M4A.WAV', '.wav')
        
        return file_info
    
    def scan_directory(self, directory, category):
        """Scan a directory for audio files."""
        dir_path = self.base_path / directory
        
        if not dir_path.exists():
            print(f"⚠️  Directory not found: {dir_path}")
            return
        
        print(f"\n🔍 Scanning {category}: {directory}")
        print(f"   Path: {dir_path}")
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(list(dir_path.rglob(f'*{ext}')))
        
        print(f"   Found {len(audio_files)} audio files")
        
        # Check each file
        for idx, file_path in enumerate(audio_files, 1):
            if idx % 50 == 0:
                print(f"   Progress: {idx}/{len(audio_files)} files checked...")
            
            file_info = self.check_audio_file(file_path)
            file_info['category'] = category
            file_info['relative_path'] = str(file_path.relative_to(self.base_path))
            self.results.append(file_info)
            
            self.summary['total_files'] += 1
            if file_info['playable']:
                self.summary['playable'] += 1
            else:
                self.summary['unplayable'] += 1
                self.summary['errors'].append({
                    'file': file_info['file_name'],
                    'error': file_info['error']
                })
    
    def fix_naming_conventions(self, dry_run=True):
        """Rename files with naming issues."""
        files_to_rename = [r for r in self.results if r['needs_rename']]
        
        print(f"\n📝 Found {len(files_to_rename)} files with naming issues")
        
        if dry_run:
            print("   (DRY RUN - no actual changes will be made)")
        
        for file_info in files_to_rename:
            old_path = Path(file_info['file_path'])
            new_name = file_info['suggested_name']
            new_path = old_path.parent / new_name
            
            print(f"   {old_path.name} → {new_name}")
            
            if not dry_run:
                try:
                    old_path.rename(new_path)
                    self.summary['renamed'] += 1
                    file_info['file_path'] = str(new_path)
                    file_info['file_name'] = new_name
                    print(f"      ✅ Renamed successfully")
                except Exception as e:
                    print(f"      ❌ Error: {e}")
    
    def generate_report(self, output_dir='validation_reports'):
        """Generate detailed validation report."""
        report_dir = self.base_path / output_dir
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results to CSV
        df = pd.DataFrame(self.results)
        csv_path = report_dir / f'audio_validation_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Detailed results saved to: {csv_path}")
        
        # Generate summary report
        report_path = report_dir / f'validation_summary_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AUDIO FILE VALIDATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Base Path: {self.base_path}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Files Scanned:    {self.summary['total_files']}\n")
            f.write(f"Playable Files:         {self.summary['playable']} ({self.summary['playable']/max(1, self.summary['total_files'])*100:.1f}%)\n")
            f.write(f"Unplayable Files:       {self.summary['unplayable']} ({self.summary['unplayable']/max(1, self.summary['total_files'])*100:.1f}%)\n")
            f.write(f"Files Renamed:          {self.summary['renamed']}\n\n")
            
            # Category breakdown
            f.write("-" * 80 + "\n")
            f.write("CATEGORY BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            category_stats = df.groupby('category').agg({
                'playable': ['sum', 'count'],
                'file_size_kb': 'mean',
                'duration': 'mean'
            }).round(2)
            f.write(str(category_stats) + "\n\n")
            
            # Unplayable files
            if self.summary['unplayable'] > 0:
                f.write("-" * 80 + "\n")
                f.write(f"UNPLAYABLE FILES ({self.summary['unplayable']} files)\n")
                f.write("-" * 80 + "\n")
                unplayable = df[df['playable'] == False][['relative_path', 'error', 'file_size_kb']]
                for idx, row in unplayable.iterrows():
                    f.write(f"\nFile: {row['relative_path']}\n")
                    f.write(f"  Size: {row['file_size_kb']:.2f} KB\n")
                    f.write(f"  Error: {row['error']}\n")
            
            # Playable file statistics
            playable_df = df[df['playable'] == True]
            if len(playable_df) > 0:
                f.write("\n" + "-" * 80 + "\n")
                f.write("PLAYABLE FILES STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Average Duration:       {playable_df['duration'].mean():.2f} seconds\n")
                f.write(f"Min Duration:           {playable_df['duration'].min():.2f} seconds\n")
                f.write(f"Max Duration:           {playable_df['duration'].max():.2f} seconds\n")
                f.write(f"Average Sample Rate:    {playable_df['sample_rate'].mean():.0f} Hz\n")
                f.write(f"Average File Size:      {playable_df['file_size_kb'].mean():.2f} KB\n")
        
        print(f"📊 Summary report saved to: {report_path}\n")
        
        # Print console summary
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"✅ Playable:   {self.summary['playable']}/{self.summary['total_files']} ({self.summary['playable']/max(1, self.summary['total_files'])*100:.1f}%)")
        print(f"❌ Unplayable: {self.summary['unplayable']}/{self.summary['total_files']} ({self.summary['unplayable']/max(1, self.summary['total_files'])*100:.1f}%)")
        if self.summary['renamed'] > 0:
            print(f"📝 Renamed:    {self.summary['renamed']} files")
        print("=" * 80)
        
        return df


def main():
    """Main execution function."""
    print("🎵 Audio File Validation & Naming Convention Tool")
    print("=" * 80)
    
    # Configuration
    base_path = r"C:\Users\Harsh Jain\Downloads\prism"
    
    # Initialize validator
    validator = AudioFileValidator(base_path)
    
    # Scan directories
    directories_to_scan = [
        ('Bonafide', 'Bonafide'),
        ('Spoofed', 'Spoofed'),
        ('1m', '1m_dataset'),
        ('M1', 'M1_speaker'),
        ('M2', 'M2_speaker'),
        ('F1', 'F1_speaker'),
        ('F2', 'F2_speaker'),
        ('F3', 'F3_speaker'),
    ]
    
    for directory, category in directories_to_scan:
        validator.scan_directory(directory, category)
    
    # Generate report
    results_df = validator.generate_report()
    
    # Check for naming issues
    print("\n" + "=" * 80)
    print("CHECKING NAMING CONVENTIONS")
    print("=" * 80)
    validator.fix_naming_conventions(dry_run=True)
    
    # Ask user if they want to apply naming fixes
    print("\n" + "=" * 80)
    response = input("\n⚠️  Do you want to apply naming fixes? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        validator.fix_naming_conventions(dry_run=False)
        print("\n✅ Naming fixes applied!")
        # Regenerate report with updated names
        validator.generate_report()
    else:
        print("\n⏭️  Skipped naming fixes (dry run only)")
    
    # Show problematic files
    if validator.summary['unplayable'] > 0:
        print("\n" + "=" * 80)
        print("⚠️  ATTENTION: Unplayable Files Detected")
        print("=" * 80)
        print(f"Found {validator.summary['unplayable']} unplayable files.")
        print("These files may have compatibility issues on Windows.")
        print("\nCommon issues:")
        print("  • .m4a files with Apple-specific codecs")
        print("  • Corrupted audio data")
        print("  • Invalid file headers")
        print("\nCheck the validation report for details.")
        print("Consider converting .m4a files to .wav format using:")
        print("  ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav")
    
    print("\n✅ Validation complete!")
    return results_df


if __name__ == "__main__":
    results = main()
