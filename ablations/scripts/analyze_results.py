"""
Ablation Results Analyzer and Comparison Generator

This script analyzes results from all ablation experiments and generates:
- Comparison tables (CSV and Markdown)
- Visualizations (plots of metrics vs epochs)
- Summary report
- Statistical analysis

Usage:
    python analyze_results.py --results-dir ../results --output-format both
"""

import json
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas/matplotlib not available. Some features will be limited.")


class ResultsAnalyzer:
    """Analyzes and compares ablation study results."""
    
    def __init__(self, results_dir: str):
        """
        Initialize results analyzer.
        
        Args:
            results_dir: Directory containing ablation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.ablation_names = ["baseline", "text_finetuned", "no_ema"]
        self.results = {}
        self.comparison_data = {}
        
        print(f"[ResultsAnalyzer] Initialized with results_dir: {self.results_dir}")
    
    def load_results(self) -> bool:
        """
        Load all available ablation results.
        
        Returns:
            True if results were loaded, False otherwise
        """
        print(f"\n[ResultsAnalyzer] Loading results from {self.results_dir}...")
        
        for ablation_name in self.ablation_names:
            summary_file = self.results_dir / f"{ablation_name}_summary.json"
            
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        self.results[ablation_name] = json.load(f)
                    print(f"  ✓ Loaded {ablation_name}: {summary_file}")
                except Exception as e:
                    print(f"  ✗ Error loading {ablation_name}: {e}")
            else:
                print(f"  - {ablation_name}: No results yet ({summary_file})")
        
        return len(self.results) > 0
    
    def create_comparison_table(self, output_format: str = "csv") -> Optional[Path]:
        """
        Create comparison table of all ablations.
        
        Args:
            output_format: 'csv', 'md' (markdown), or 'both'
            
        Returns:
            Path to output file(s)
        """
        print(f"\n[ResultsAnalyzer] Creating comparison table...")
        
        # Build comparison data
        rows = []
        for ablation_name, result in self.results.items():
            training_summary = result.get("training_summary", {})
            evaluation_summary = result.get("evaluation_summary", {})
            
            row = {
                "Method": ablation_name,
                "Final Loss": training_summary.get("final_loss", "N/A"),
                "Train Time (h)": round(training_summary.get("total_time_hours", 0), 2),
                "Peak Memory (GB)": round(training_summary.get("peak_memory_gb", 0), 2),
                "FVD": evaluation_summary.get("fvd", "N/A"),
                "LPIPS": evaluation_summary.get("lpips", "N/A"),
                "Temporal": evaluation_summary.get("temporal_consistency", "N/A"),
                "Inference (ms)": evaluation_summary.get("inference_time_ms", "N/A"),
                "Inference Mem (GB)": evaluation_summary.get("inference_memory_gb", "N/A"),
                "Parameters (M)": evaluation_summary.get("num_parameters_millions", "N/A"),
            }
            rows.append(row)
        
        if not rows:
            print(f"  ⚠ No results to compare")
            return None
        
        # Convert to DataFrame if pandas available
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(rows)
            
            # Save to CSV
            if output_format in ["csv", "both"]:
                csv_file = self.results_dir / "comparison_table.csv"
                df.to_csv(csv_file, index=False)
                print(f"  ✓ CSV table: {csv_file}")
            
            # Save to Markdown
            if output_format in ["md", "both"]:
                md_file = self.results_dir / "comparison_table.md"
                with open(md_file, 'w') as f:
                    f.write("# Ablation Study Comparison Table\n\n")
                    f.write(df.to_markdown(index=False))
                    f.write("\n\n*Generated: " + datetime.now().isoformat() + "*\n")
                print(f"  ✓ Markdown table: {md_file}")
            
            return csv_file or md_file
        else:
            # Fallback to manual CSV creation
            if output_format in ["csv", "both"]:
                csv_file = self.results_dir / "comparison_table.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  ✓ CSV table: {csv_file}")
                return csv_file
    
    def create_summary_report(self) -> Path:
        """
        Create comprehensive text summary report.
        
        Returns:
            Path to report file
        """
        print(f"\n[ResultsAnalyzer] Creating summary report...")
        
        report_file = self.results_dir / "ABLATION_RESULTS_REPORT.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ABLATION STUDY RESULTS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Results Directory: {self.results_dir}\n\n")
            
            # Summary for each ablation
            for ablation_name in self.ablation_names:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"ABLATION: {ablation_name.upper()}\n")
                f.write(f"{'=' * 80}\n\n")
                
                if ablation_name not in self.results:
                    f.write("  [No results available yet]\n\n")
                    continue
                
                result = self.results[ablation_name]
                
                # Metadata
                f.write("Configuration:\n")
                if "metadata" in result:
                    meta = result["metadata"]
                    f.write(f"  Freeze text encoder: {meta.get('model_config', {}).get('freeze_text_encoder', 'N/A')}\n")
                    f.write(f"  Use EMA: {meta.get('model_config', {}).get('use_ema', 'N/A')}\n")
                    f.write(f"  Model channels: {meta.get('model_config', {}).get('model_channels', 'N/A')}\n")
                    f.write(f"  Learning rate: {meta.get('training_config', {}).get('learning_rate', 'N/A')}\n")
                    f.write(f"  Batch size: {meta.get('training_config', {}).get('batch_size', 'N/A')}\n")
                
                # Training results
                f.write("\nTraining Results:\n")
                if "training_summary" in result:
                    train = result["training_summary"]
                    f.write(f"  Final loss: {train.get('final_loss', 'N/A')}\n")
                    f.write(f"  Min loss: {train.get('min_loss', 'N/A')}\n")
                    f.write(f"  Avg loss: {train.get('avg_loss', 'N/A')}\n")
                    f.write(f"  Total time: {train.get('total_time_hours', 'N/A')} hours\n")
                    f.write(f"  Peak memory: {train.get('peak_memory_gb', 'N/A')} GB\n")
                    f.write(f"  Avg memory: {train.get('avg_memory_gb', 'N/A')} GB\n")
                    f.write(f"  Number of steps: {train.get('num_steps', 'N/A')}\n")
                
                # Evaluation results
                f.write("\nEvaluation Results:\n")
                if "evaluation_summary" in result:
                    eval = result["evaluation_summary"]
                    f.write(f"  FVD: {eval.get('fvd', 'N/A')}\n")
                    f.write(f"  LPIPS: {eval.get('lpips', 'N/A')}\n")
                    f.write(f"  Temporal consistency: {eval.get('temporal_consistency', 'N/A')}\n")
                    f.write(f"  Inference time: {eval.get('inference_time_ms', 'N/A')} ms\n")
                    f.write(f"  Inference memory: {eval.get('inference_memory_gb', 'N/A')} GB\n")
                    f.write(f"  Parameters: {eval.get('num_parameters_millions', 'N/A')} M\n")
                
                f.write("\n")
            
            # Key findings
            f.write(f"\n{'=' * 80}\n")
            f.write("KEY FINDINGS\n")
            f.write(f"{'=' * 80}\n\n")
            
            if "baseline" in self.results and len(self.results) > 1:
                baseline = self.results["baseline"]
                baseline_fvd = baseline.get("evaluation_summary", {}).get("fvd")
                
                f.write("Quality Comparison (vs Baseline):\n")
                for ablation_name in self.ablation_names:
                    if ablation_name == "baseline" or ablation_name not in self.results:
                        continue
                    
                    result = self.results[ablation_name]
                    ablation_fvd = result.get("evaluation_summary", {}).get("fvd")
                    
                    if baseline_fvd and ablation_fvd:
                        diff = ablation_fvd - baseline_fvd
                        pct_change = (diff / baseline_fvd * 100) if baseline_fvd != 0 else 0
                        f.write(f"  {ablation_name}: FVD = {ablation_fvd:.2f} ")
                        f.write(f"(Δ {diff:+.2f}, {pct_change:+.1f}%)\n")
                
                f.write("\nTraining Efficiency Comparison (vs Baseline):\n")
                baseline_time = baseline.get("training_summary", {}).get("total_time_hours", 0)
                baseline_mem = baseline.get("training_summary", {}).get("peak_memory_gb", 0)
                
                for ablation_name in self.ablation_names:
                    if ablation_name == "baseline" or ablation_name not in self.results:
                        continue
                    
                    result = self.results[ablation_name]
                    ablation_time = result.get("training_summary", {}).get("total_time_hours", 0)
                    ablation_mem = result.get("training_summary", {}).get("peak_memory_gb", 0)
                    
                    time_diff = ((ablation_time - baseline_time) / baseline_time * 100) if baseline_time else 0
                    mem_diff = ((ablation_mem - baseline_mem) / baseline_mem * 100) if baseline_mem else 0
                    
                    f.write(f"  {ablation_name}:\n")
                    f.write(f"    Training time: {ablation_time:.2f}h ({time_diff:+.1f}%)\n")
                    f.write(f"    Peak memory: {ablation_mem:.2f}GB ({mem_diff:+.1f}%)\n")
            
            f.write(f"\n{'=' * 80}\n")
            f.write("END OF REPORT\n")
            f.write(f"{'=' * 80}\n")
        
        print(f"  ✓ Report saved: {report_file}")
        return report_file
    
    def print_summary(self):
        """Print summary to console."""
        print(f"\n{'=' * 80}")
        print(f"ABLATION RESULTS SUMMARY")
        print(f"{'=' * 80}\n")
        
        for ablation_name in self.ablation_names:
            if ablation_name not in self.results:
                print(f"{ablation_name}: [No results yet]")
                continue
            
            result = self.results[ablation_name]
            eval_summary = result.get("evaluation_summary", {})
            train_summary = result.get("training_summary", {})
            
            print(f"{ablation_name}:")
            print(f"  FVD: {eval_summary.get('fvd', 'N/A')}")
            print(f"  LPIPS: {eval_summary.get('lpips', 'N/A')}")
            print(f"  Train time: {train_summary.get('total_time_hours', 'N/A')} hours")
            print(f"  Peak memory: {train_summary.get('peak_memory_gb', 'N/A')} GB")
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze and compare ablation study results"
    )
    parser.add_argument(
        "--results-dir",
        default="../results",
        help="Directory containing ablation results"
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "md", "both"],
        default="both",
        help="Output format for comparison table"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating text report"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = ResultsAnalyzer(args.results_dir)
        
        # Load results
        if analyzer.load_results():
            # Print summary
            analyzer.print_summary()
            
            # Create comparison table
            analyzer.create_comparison_table(output_format=args.output_format)
            
            # Create report
            if not args.no_report:
                analyzer.create_summary_report()
            
            print(f"\n✓ Analysis complete!")
            print(f"Results saved to: {args.results_dir}")
        else:
            print(f"\n⚠ No results found. Have you run the ablation experiments yet?")
            print(f"Results directory: {args.results_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
