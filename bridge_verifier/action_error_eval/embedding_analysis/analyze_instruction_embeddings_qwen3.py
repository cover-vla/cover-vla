#!/usr/bin/env python3
"""
Script to analyze the relationship between instruction embedding similarity and action errors.

This script:
1. Loads the JSON data containing instructions and action predictions
2. Uses Qwen3-Embedding-8B to embed both "instruction" and "original_instruction" 
3. Calculates cosine similarity between embeddings
4. Investigates correlation between embedding similarity and openvla_nrmse (action error)
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import torch
from vllm import LLM
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class InstructionEmbeddingAnalyzer:
    def __init__(self, model_name='Qwen/Qwen3-Embedding-8B'):
        """Initialize Qwen3 embedding model."""
        print(f"Loading Qwen3 embedding model: {model_name}")
        self.model = LLM(model=model_name, task="embed")
        self.model_name = model_name
        print(f"Model loaded: {self.model_name}")
    
    def get_qwen3_embedding(self, text):
        """Get Qwen3 embedding for a single text."""
        outputs = self.model.embed([text])
        embedding = torch.tensor([o.outputs.embedding for o in outputs])
        return embedding.squeeze().numpy()
    
    def get_embeddings_batch(self, texts, batch_size=1000):
        """Get Qwen3 embeddings for a batch of texts."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Get embeddings for the batch
            outputs = self.model.embed(batch_texts)
            batch_embeddings = torch.tensor([o.outputs.embedding for o in outputs])
            
            embeddings.append(batch_embeddings.numpy())
        
        return np.vstack(embeddings)
    
    def load_data(self, json_file):
        """Load and parse the JSON data."""
        print(f"Loading data from {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract results
        results = data['results']
        print(f"Loaded {len(results)} samples")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Filter out samples where instruction == original_instruction (no rephrasing)
        df_rephrased = df[df['instruction'] != df['original_instruction']].copy()
        print(f"Found {len(df_rephrased)} samples with rephrased instructions")
        
        return df, df_rephrased
    
    def calculate_similarities(self, df):
        """Calculate cosine similarities between instruction embeddings."""
        print("Generating Qwen3 embeddings for instructions...")
        instructions = df['instruction'].tolist()
        original_instructions = df['original_instruction'].tolist()
        
        # Get embeddings
        instruction_embeddings = self.get_embeddings_batch(instructions)
        original_embeddings = self.get_embeddings_batch(original_instructions)
        
        # Calculate cosine similarities
        print("Calculating cosine similarities...")
        similarities = []
        for i in range(len(instruction_embeddings)):
            sim = cosine_similarity([instruction_embeddings[i]], [original_embeddings[i]])[0][0]
            similarities.append(sim)
        
        df['embedding_similarity'] = similarities
        return df
    
    def analyze_correlation(self, df):
        """Analyze correlation between embedding similarity and action error."""
        print("\nAnalyzing correlations...")
        
        # Calculate correlations
        pearson_corr, pearson_p = pearsonr(df['embedding_similarity'], df['openvla_nrmse'])
        spearman_corr, spearman_p = spearmanr(df['embedding_similarity'], df['openvla_nrmse'])
        
        print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
        print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        print(f"Embedding similarity - Mean: {df['embedding_similarity'].mean():.4f}, Std: {df['embedding_similarity'].std():.4f}")
        print(f"OpenVLA NRMSE - Mean: {df['openvla_nrmse'].mean():.4f}, Std: {df['openvla_nrmse'].std():.4f}")
        
        return pearson_corr, pearson_p, spearman_corr, spearman_p
    
    def create_visualizations(self, df, output_dir='.'):
        """Create visualizations of the analysis."""
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Instruction Embedding Similarity vs Action Error Analysis (Qwen3)', fontsize=16)
        
        # 1. Scatter plot: Embedding similarity vs NRMSE
        axes[0, 0].scatter(df['embedding_similarity'], df['openvla_nrmse'], alpha=0.6)
        axes[0, 0].set_xlabel('Embedding Similarity (Cosine)')
        axes[0, 0].set_ylabel('OpenVLA NRMSE (Action Error)')
        axes[0, 0].set_title('Embedding Similarity vs Action Error')
        
        # Add trend line
        z = np.polyfit(df['embedding_similarity'], df['openvla_nrmse'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(df['embedding_similarity'], p(df['embedding_similarity']), "r--", alpha=0.8)
        
        # 2. Distribution of embedding similarities
        axes[0, 1].hist(df['embedding_similarity'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Embedding Similarity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Embedding Similarities')
        axes[0, 1].axvline(df['embedding_similarity'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["embedding_similarity"].mean():.3f}')
        axes[0, 1].legend()
        
        # 3. Distribution of NRMSE values
        axes[1, 0].hist(df['openvla_nrmse'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('OpenVLA NRMSE')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Action Errors (NRMSE)')
        axes[1, 0].axvline(df['openvla_nrmse'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["openvla_nrmse"].mean():.3f}')
        axes[1, 0].legend()
        
        # 4. Binned analysis
        # Create bins based on embedding similarity
        df['similarity_bin'] = pd.cut(df['embedding_similarity'], bins=10, labels=False)
        bin_stats = df.groupby('similarity_bin')['openvla_nrmse'].agg(['mean', 'std', 'count'])
        
        bin_centers = []
        for i in range(10):
            bin_data = df[df['similarity_bin'] == i]
            if len(bin_data) > 0:
                bin_centers.append(bin_data['embedding_similarity'].mean())
            else:
                bin_centers.append(np.nan)
        
        valid_bins = ~np.isnan(bin_centers)
        axes[1, 1].errorbar(np.array(bin_centers)[valid_bins], 
                           bin_stats['mean'].values[valid_bins],
                           yerr=bin_stats['std'].values[valid_bins],
                           fmt='o-', capsize=5)
        axes[1, 1].set_xlabel('Embedding Similarity (Bin Centers)')
        axes[1, 1].set_ylabel('Mean OpenVLA NRMSE')
        axes[1, 1].set_title('Binned Analysis: Similarity vs Mean Error')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/qwen3_embedding_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional detailed analysis plot
        fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create a more detailed scatter plot with color coding
        scatter = ax.scatter(df['embedding_similarity'], df['openvla_nrmse'], 
                           c=df['monkey_verifier_score'], cmap='viridis', alpha=0.6)
        ax.set_xlabel('Embedding Similarity (Cosine)')
        ax.set_ylabel('OpenVLA NRMSE (Action Error)')
        ax.set_title('Embedding Similarity vs Action Error (Qwen3)\n(Color: Monkey Verifier Score)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Monkey Verifier Score')
        
        # Add correlation info as text
        pearson_corr, pearson_p, spearman_corr, spearman_p = self.analyze_correlation(df)
        textstr = f'Pearson r = {pearson_corr:.3f} (p = {pearson_p:.3f})\nSpearman ρ = {spearman_corr:.3f} (p = {spearman_p:.3f})'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/qwen3_detailed_embedding_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, df, output_file='qwen3_embedding_analysis_report.txt'):
        """Generate a comprehensive text report."""
        print(f"Generating report: {output_file}")
        
        pearson_corr, pearson_p, spearman_corr, spearman_p = self.analyze_correlation(df)
        
        with open(output_file, 'w') as f:
            f.write("INSTRUCTION EMBEDDING SIMILARITY vs ACTION ERROR ANALYSIS (Qwen3)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("MODEL INFORMATION:\n")
            f.write(f"Embedding Model: {self.model_name}\n")
            f.write(f"Embedding Dimension: {len(df['embedding_similarity']) if len(df) > 0 else 'N/A'}\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write(f"Total samples analyzed: {len(df)}\n")
            f.write(f"Samples with rephrased instructions: {len(df[df['instruction'] != df['original_instruction']])}\n\n")
            
            f.write("EMBEDDING SIMILARITY STATISTICS:\n")
            f.write(f"Mean similarity: {df['embedding_similarity'].mean():.4f}\n")
            f.write(f"Std deviation: {df['embedding_similarity'].std():.4f}\n")
            f.write(f"Min similarity: {df['embedding_similarity'].min():.4f}\n")
            f.write(f"Max similarity: {df['embedding_similarity'].max():.4f}\n")
            f.write(f"Median similarity: {df['embedding_similarity'].median():.4f}\n\n")
            
            f.write("ACTION ERROR (OPENVLA_NRMSE) STATISTICS:\n")
            f.write(f"Mean NRMSE: {df['openvla_nrmse'].mean():.4f}\n")
            f.write(f"Std deviation: {df['openvla_nrmse'].std():.4f}\n")
            f.write(f"Min NRMSE: {df['openvla_nrmse'].min():.4f}\n")
            f.write(f"Max NRMSE: {df['openvla_nrmse'].max():.4f}\n")
            f.write(f"Median NRMSE: {df['openvla_nrmse'].median():.4f}\n\n")
            
            f.write("CORRELATION ANALYSIS:\n")
            f.write(f"Pearson correlation coefficient: {pearson_corr:.4f}\n")
            f.write(f"Pearson p-value: {pearson_p:.4f}\n")
            f.write(f"Spearman correlation coefficient: {spearman_corr:.4f}\n")
            f.write(f"Spearman p-value: {spearman_p:.4f}\n\n")
            
            f.write("INTERPRETATION:\n")
            if pearson_p < 0.05:
                if pearson_corr < 0:
                    f.write("✓ SIGNIFICANT NEGATIVE CORRELATION: Higher embedding similarity is associated with LOWER action errors.\n")
                    f.write("  This suggests that when rephrased instructions are more similar to original instructions,\n")
                    f.write("  the model performs better (lower NRMSE).\n")
                else:
                    f.write("✗ SIGNIFICANT POSITIVE CORRELATION: Higher embedding similarity is associated with HIGHER action errors.\n")
                    f.write("  This is counterintuitive and may warrant further investigation.\n")
            else:
                f.write("○ NO SIGNIFICANT CORRELATION: Embedding similarity does not significantly predict action error.\n")
                f.write("  The relationship between instruction similarity and model performance is not clear.\n")
            
            f.write(f"\nCorrelation strength: {abs(pearson_corr):.3f} - ")
            if abs(pearson_corr) < 0.1:
                f.write("Very weak\n")
            elif abs(pearson_corr) < 0.3:
                f.write("Weak\n")
            elif abs(pearson_corr) < 0.5:
                f.write("Moderate\n")
            elif abs(pearson_corr) < 0.7:
                f.write("Strong\n")
            else:
                f.write("Very strong\n")
            
            # Top and bottom examples
            f.write("\nEXAMPLES:\n")
            f.write("Top 5 most similar instruction pairs (highest embedding similarity):\n")
            top_similar = df.nlargest(5, 'embedding_similarity')[['instruction', 'original_instruction', 'embedding_similarity', 'openvla_nrmse']]
            for idx, row in top_similar.iterrows():
                f.write(f"  Similarity: {row['embedding_similarity']:.3f}, NRMSE: {row['openvla_nrmse']:.3f}\n")
                f.write(f"    Original: '{row['original_instruction']}'\n")
                f.write(f"    Rephrased: '{row['instruction']}'\n\n")
            
            f.write("Top 5 least similar instruction pairs (lowest embedding similarity):\n")
            bottom_similar = df.nsmallest(5, 'embedding_similarity')[['instruction', 'original_instruction', 'embedding_similarity', 'openvla_nrmse']]
            for idx, row in bottom_similar.iterrows():
                f.write(f"  Similarity: {row['embedding_similarity']:.3f}, NRMSE: {row['openvla_nrmse']:.3f}\n")
                f.write(f"    Original: '{row['original_instruction']}'\n")
                f.write(f"    Rephrased: '{row['instruction']}'\n\n")
        
        print(f"Report saved to {output_file}")
    
    def compare_with_bert_results(self, df, bert_results_file=None, output_dir='.'):
        """Compare Qwen3 results with BERT results if available."""
        if bert_results_file and os.path.exists(bert_results_file):
            print("Comparing with BERT results...")
            bert_df = pd.read_csv(bert_results_file)
            
            # Merge on common columns to compare similarities
            comparison_df = df.merge(bert_df[['instruction', 'original_instruction', 'embedding_similarity']], 
                                   on=['instruction', 'original_instruction'], 
                                   suffixes=('_qwen3', '_bert'))
            
            if len(comparison_df) > 0:
                # Calculate correlation between BERT and Qwen3 similarities
                bert_qwen3_corr = pearsonr(comparison_df['embedding_similarity_bert'], 
                                         comparison_df['embedding_similarity_qwen3'])
                
                print(f"\nBERT vs Qwen3 Similarity Correlation:")
                print(f"Pearson correlation: {bert_qwen3_corr[0]:.4f} (p-value: {bert_qwen3_corr[1]:.4f})")
                
                # Create comparison plot
                plt.figure(figsize=(10, 6))
                plt.scatter(comparison_df['embedding_similarity_bert'], 
                           comparison_df['embedding_similarity_qwen3'], alpha=0.6)
                plt.xlabel('BERT Embedding Similarity')
                plt.ylabel('Qwen3 Embedding Similarity')
                plt.title('BERT vs Qwen3 Embedding Similarities')
                
                # Add diagonal line for reference
                min_val = min(comparison_df['embedding_similarity_bert'].min(), 
                            comparison_df['embedding_similarity_qwen3'].min())
                max_val = max(comparison_df['embedding_similarity_bert'].max(), 
                            comparison_df['embedding_similarity_qwen3'].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect correlation')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/bert_vs_qwen3_comparison.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                return comparison_df
        else:
            print("No BERT results file found for comparison.")
            return None

def main():
    """Main analysis pipeline."""
    # Configuration
    json_file = '/root/validate_clip_verifier/bridge_monkey_verifier_scores_20250810_002507.json'
    
    # Initialize analyzer
    analyzer = InstructionEmbeddingAnalyzer()
    
    # Load data
    df_all, df_rephrased = analyzer.load_data(json_file)
    
    # Focus on rephrased instructions for the main analysis
    if len(df_rephrased) == 0:
        print("No rephrased instructions found. Analyzing all data...")
        df_analysis = df_all
    else:
        df_analysis = df_rephrased
    
    # Calculate similarities
    df_analysis = analyzer.calculate_similarities(df_analysis)
    
    # Analyze correlations
    analyzer.analyze_correlation(df_analysis)
    
    # Create visualizations
    analyzer.create_visualizations(df_analysis)
    
    # Generate report
    analyzer.generate_report(df_analysis)
    
    # Compare with BERT results if available
    analyzer.compare_with_bert_results(df_analysis, 'embedding_analysis_data.csv')
    
    # Save processed data
    df_analysis.to_csv('qwen3_embedding_analysis_data.csv', index=False)
    print("Processed data saved to qwen3_embedding_analysis_data.csv")
    
    print("\nQwen3 embedding analysis complete!")

if __name__ == "__main__":
    main()
