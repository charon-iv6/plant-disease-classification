from visualization_base import VisualizationBase
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import List, Dict, Optional
import logging
import torch
import torch.nn as nn

class SubmissionVisualizer(VisualizationBase):
    """Visualization tools for submission analysis and comparison"""
    
    def __init__(self, save_dir: str = 'assets/submission_viz'):
        super().__init__(save_dir)
        self.logger = logging.getLogger(__name__)
        
    def compare_submissions(self, submission_files: List[str],
                          metadata_files: Optional[List[str]] = None) -> None:
        """Compare multiple submission files"""
        try:
            # Load submissions
            submissions = []
            for file in submission_files:
                df = pd.read_csv(file)
                submission_id = Path(file).stem
                df['submission_id'] = submission_id
                submissions.append(df)
            
            combined_df = pd.concat(submissions)
            
            # Load metadata if provided
            metadata = {}
            if metadata_files:
                for file in metadata_files:
                    with open(file, 'r') as f:
                        metadata[Path(file).stem] = json.load(f)
            
            # Generate visualizations
            self._plot_prediction_distribution(combined_df)
            self._plot_submission_agreement(submissions)
            if metadata:
                self._plot_performance_comparison(metadata)
            
            # Generate comparison report
            self._generate_comparison_report(combined_df, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to compare submissions: {str(e)}")
    
    def _plot_prediction_distribution(self, combined_df: pd.DataFrame) -> None:
        """Plot distribution of predictions across submissions"""
        # Create distribution plot
        plt.figure(figsize=(15, 8))
        
        for submission_id in combined_df['submission_id'].unique():
            subset = combined_df[combined_df['submission_id'] == submission_id]
            sns.kdeplot(data=subset['prediction'], label=submission_id)
        
        plt.title('Distribution of Predictions Across Submissions')
        plt.xlabel('Predicted Class')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_dir / 'prediction_distribution.png')
        plt.close()
        
        # Create interactive version
        fig = go.Figure()
        
        for submission_id in combined_df['submission_id'].unique():
            subset = combined_df[combined_df['submission_id'] == submission_id]
            fig.add_trace(
                go.Violin(y=subset['prediction'],
                         name=submission_id,
                         box_visible=True,
                         meanline_visible=True)
            )
        
        fig.update_layout(
            title='Distribution of Predictions Across Submissions',
            yaxis_title='Predicted Class',
            showlegend=True
        )
        
        fig.write_html(self.save_dir / 'prediction_distribution.html')
    
    def _plot_submission_agreement(self, submissions: List[pd.DataFrame]) -> None:
        """Plot agreement between different submissions"""
        # Create agreement matrix
        n_submissions = len(submissions)
        agreement_matrix = np.zeros((n_submissions, n_submissions))
        
        for i in range(n_submissions):
            for j in range(n_submissions):
                agreement = np.mean(
                    submissions[i]['prediction'].values == 
                    submissions[j]['prediction'].values
                )
                agreement_matrix[i, j] = agreement
        
        # Plot agreement matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(agreement_matrix,
                   annot=True,
                   fmt='.3f',
                   xticklabels=[df['submission_id'].iloc[0] for df in submissions],
                   yticklabels=[df['submission_id'].iloc[0] for df in submissions],
                   cmap='YlOrRd')
        plt.title('Submission Agreement Matrix')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'submission_agreement.png')
        plt.close()
        
        # Create interactive version
        fig = go.Figure(data=go.Heatmap(
            z=agreement_matrix,
            x=[df['submission_id'].iloc[0] for df in submissions],
            y=[df['submission_id'].iloc[0] for df in submissions],
            text=agreement_matrix.round(3),
            texttemplate='%{text}',
            textfont={'size': 12},
            colorscale='YlOrRd'
        ))
        
        fig.update_layout(
            title='Submission Agreement Matrix',
            xaxis_title='Submission ID',
            yaxis_title='Submission ID'
        )
        
        fig.write_html(self.save_dir / 'submission_agreement.html')
    
    def _plot_performance_comparison(self, metadata: Dict) -> None:
        """Plot performance metrics comparison"""
        # Extract metrics
        metrics = {
            'inference_time': [],
            'memory_usage': [],
            'submission_id': []
        }
        
        for submission_id, data in metadata.items():
            metrics['inference_time'].append(data['inference']['avg_inference_time'])
            metrics['memory_usage'].append(data['inference']['memory_peak'])
            metrics['submission_id'].append(submission_id)
        
        # Create comparison plots
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Inference Time', 'Memory Usage'))
        
        # Inference time
        fig.add_trace(
            go.Bar(x=metrics['submission_id'],
                  y=metrics['inference_time'],
                  name='Inference Time'),
            row=1, col=1
        )
        
        # Memory usage
        fig.add_trace(
            go.Bar(x=metrics['submission_id'],
                  y=metrics['memory_usage'],
                  name='Memory Usage'),
            row=1, col=2
        )
        
        fig.update_layout(height=500, title_text="Performance Comparison")
        fig.write_html(self.save_dir / 'performance_comparison.html')
        
        # Create static version
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(metrics['submission_id'], metrics['inference_time'])
        plt.title('Inference Time')
        plt.xlabel('Submission ID')
        plt.ylabel('Time (s)')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.bar(metrics['submission_id'], metrics['memory_usage'])
        plt.title('Memory Usage')
        plt.xlabel('Submission ID')
        plt.ylabel('Memory (GB)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'performance_comparison.png')
        plt.close()
    
    def _generate_comparison_report(self, combined_df: pd.DataFrame,
                                  metadata: Optional[Dict] = None) -> None:
        """Generate comprehensive comparison report"""
        with open(self.save_dir / 'comparison_report.md', 'w') as f:
            f.write("# Submission Comparison Report\n\n")
            
            # Basic statistics
            f.write("## Basic Statistics\n\n")
            f.write("### Sample Counts\n")
            for submission_id in combined_df['submission_id'].unique():
                count = len(combined_df[combined_df['submission_id'] == submission_id])
                f.write(f"- {submission_id}: {count:,} samples\n")
            
            f.write("\n### Class Distribution\n")
            for submission_id in combined_df['submission_id'].unique():
                subset = combined_df[combined_df['submission_id'] == submission_id]
                dist = subset['prediction'].value_counts().to_dict()
                f.write(f"\n**{submission_id}**\n")
                for class_idx, count in dist.items():
                    class_name = self.class_names[int(class_idx)]
                    f.write(f"- {class_name}: {count:,} samples ({count/len(subset)*100:.1f}%)\n")
            
            # Agreement analysis
            f.write("\n## Agreement Analysis\n")
            submission_pairs = [
                (s1, s2) for i, s1 in enumerate(combined_df['submission_id'].unique())
                for s2 in list(combined_df['submission_id'].unique())[i+1:]
            ]
            
            for s1, s2 in submission_pairs:
                df1 = combined_df[combined_df['submission_id'] == s1]
                df2 = combined_df[combined_df['submission_id'] == s2]
                agreement = np.mean(df1['prediction'].values == df2['prediction'].values)
                f.write(f"\n### {s1} vs {s2}\n")
                f.write(f"- Overall agreement: {agreement*100:.1f}%\n")
                
                # Class-wise agreement
                f.write("- Class-wise agreement:\n")
                for class_idx in range(len(self.class_names)):
                    mask = (df1['prediction'] == class_idx) | (df2['prediction'] == class_idx)
                    if mask.any():
                        class_agreement = np.mean(
                            df1.loc[mask, 'prediction'].values == 
                            df2.loc[mask, 'prediction'].values
                        )
                        f.write(f"  - {self.class_names[class_idx]}: {class_agreement*100:.1f}%\n")
            
            # Performance comparison
            if metadata:
                f.write("\n## Performance Comparison\n\n")
                f.write("| Submission | Inference Time (s) | Memory Usage (GB) |\n")
                f.write("|------------|-------------------|------------------|\n")
                
                for submission_id, data in metadata.items():
                    inf_time = data['inference']['avg_inference_time']
                    memory = data['inference']['memory_peak']
                    f.write(f"| {submission_id} | {inf_time:.3f} | {memory:.1f} |\n")
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            f.write("Based on the analysis above:\n\n")
            
            # Add recommendations based on agreement patterns
            high_agreement_pairs = [
                (s1, s2) for s1, s2 in submission_pairs
                if np.mean(
                    combined_df[combined_df['submission_id'] == s1]['prediction'].values ==
                    combined_df[combined_df['submission_id'] == s2]['prediction'].values
                ) > 0.9
            ]
            
            if high_agreement_pairs:
                f.write("1. High Agreement Pairs:\n")
                for s1, s2 in high_agreement_pairs:
                    f.write(f"   - {s1} and {s2} show very similar predictions. ")
                    f.write("Consider diversifying these approaches.\n")
            
            if metadata:
                # Add recommendations based on performance
                inf_times = {
                    sid: data['inference']['avg_inference_time']
                    for sid, data in metadata.items()
                }
                slowest = max(inf_times.items(), key=lambda x: x[1])[0]
                
                f.write("\n2. Performance Optimization:\n")
                f.write(f"   - {slowest} has the longest inference time. ")
                f.write("Consider optimizing this submission for better performance.\n")
    
    def plot_model_architecture(self, model: nn.Module) -> None:
        """Not implemented for submission visualizer"""
        self.logger.warning("Model architecture visualization not implemented for submission visualizer")
    
    def plot_feature_maps(self, model: nn.Module,
                         sample_input: torch.Tensor,
                         layer_names: Optional[List[str]] = None) -> None:
        """Not implemented for submission visualizer"""
        self.logger.warning("Feature map visualization not implemented for submission visualizer") 