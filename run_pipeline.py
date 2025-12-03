"""
Samsung Prism - Audio Quality Pipeline Runner
=============================================

Production-grade audio scoring pipeline for Punjabi speech dataset.

Usage:
    python run_pipeline.py                    # Run with defaults
    python run_pipeline.py --dataset ./data   # Specify dataset path
    python run_pipeline.py --workers 8        # Use 8 workers
    python run_pipeline.py --report-only results.csv  # Generate report from existing CSV

Author: Samsung Prism Team
Version: 1.0.0
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import PipelineConfig
from pipeline.orchestrator import PipelineOrchestrator, run_pipeline
from pipeline.reporting import QualityReporter, generate_report


def setup_logging(output_dir: str, verbose: bool = True):
    """Configure logging for the pipeline."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return str(log_file)


def main():
    parser = argparse.ArgumentParser(
        description="Samsung Prism Audio Quality Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on current directory
  python run_pipeline.py
  
  # Run on specific dataset with 8 workers
  python run_pipeline.py --dataset ./my_audio_data --workers 8
  
  # Generate report from existing results
  python run_pipeline.py --report-only pipeline_output/latest_results.csv
  
  # Quick validation test (first 10 files)
  python run_pipeline.py --test
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='.',
        help='Path to dataset root directory (default: current directory)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='pipeline_output',
        help='Output directory for results (default: pipeline_output)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--report-only',
        type=str,
        metavar='CSV_FILE',
        help='Generate report from existing results CSV (skip processing)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run quick validation test on first 10 files'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip report generation after processing'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.output, args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("Samsung Prism Audio Quality Pipeline")
    logger.info("=" * 70)
    
    # Report-only mode
    if args.report_only:
        logger.info(f"Generating report from: {args.report_only}")
        report_dir = Path(args.output) / "reports"
        generate_report(args.report_only, str(report_dir))
        logger.info(f"Report generated in: {report_dir}")
        return 0
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return 1
    
    logger.info(f"Dataset: {dataset_path.absolute()}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Log file: {log_file}")
    
    # Configure pipeline
    config = PipelineConfig()
    config.n_workers = args.workers
    config.verbose = args.verbose
    
    # Run pipeline
    try:
        orchestrator = PipelineOrchestrator(config)
        
        if args.test:
            logger.info("Running quick validation test...")
            # For test mode, we'll run the full pipeline but it will naturally
            # process fewer files if the dataset is small
        
        df = orchestrator.run(
            str(dataset_path),
            args.output,
            args.workers
        )
        
        if df.empty:
            logger.warning("No results generated!")
            return 1
        
        logger.info(f"Processed {len(df)} files")
        
        # Generate reports
        if not args.no_report:
            logger.info("Generating reports...")
            report_dir = Path(args.output) / "reports"
            reporter = QualityReporter(df, str(report_dir))
            reporter.generate_full_report()
            logger.info(f"Reports saved to: {report_dir}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)
        print(f"Total files: {len(df)}")
        print(f"Successful: {df['success'].sum()}")
        print(f"Failed: {(~df['success']).sum()}")
        
        if 'pesq' in df.columns:
            pesq_valid = df['pesq'].dropna()
            if len(pesq_valid) > 0:
                print(f"\nPESQ: {pesq_valid.mean():.3f} ± {pesq_valid.std():.3f}")
        
        if 'snr_global_db' in df.columns:
            snr_valid = df['snr_global_db'].dropna()
            if len(snr_valid) > 0:
                print(f"SNR:  {snr_valid.mean():.1f} ± {snr_valid.std():.1f} dB")
        
        print(f"\nResults: {args.output}/latest_results.csv")
        print(f"Reports: {args.output}/reports/")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
