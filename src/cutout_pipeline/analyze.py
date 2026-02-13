"""
Audio Cutout Analysis Module

This module analyzes transcription files to detect:
1. Unusual silences
2. Grammar issues
3. Empty acknowledgments

Outputs:
- CSV with per-turn analysis
- JSON report with statistics and cutouts
- 4 PNG visualization graphs
"""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import warnings

from . import config

warnings.filterwarnings('ignore')

# Global variables for AI models (loaded once)
_embed_model = None
_fluency_tokenizer = None
_fluency_model = None


def load_models():
    """Load AI models for semantic and fluency analysis."""
    global _embed_model, _fluency_tokenizer, _fluency_model
    
    if _embed_model is None:
        print("⏳ Loading AI Models... (this may take ~30 seconds)")
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
        _fluency_tokenizer = AutoTokenizer.from_pretrained(config.FLUENCY_MODEL)
        _fluency_model = AutoModelForMaskedLM.from_pretrained(config.FLUENCY_MODEL)
        print("✅ Models loaded")


def calculate_fluency(text):
    """Calculate fluency score using masked language model."""
    if not text or len(text.strip()) == 0:
        return 100.0
    
    inputs = _fluency_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _fluency_model(**inputs, labels=inputs["input_ids"])
    
    return outputs.loss.item()


def get_outliers(data, threshold):
    """Detect outliers using Modified Z-score (MAD)."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    if mad == 0:
        return np.array([False] * len(data))
    
    modified_z_scores = 0.6745 * (data - median) / mad
    return modified_z_scores > threshold


def merge_turns(segments):
    """Merge consecutive segments from the same speaker."""
    if not segments:
        return []
    
    merged = []
    curr = segments[0].copy()
    
    for next_seg in segments[1:]:
        if next_seg['role'] == curr['role']:
            curr['text'] += " " + next_seg['text']
            curr['end'] = next_seg['end']
        else:
            merged.append(curr)
            curr = next_seg.copy()
    
    merged.append(curr)
    return merged


def save_graphs(df, cutouts, stats, output_dir, base_name):
    """Generate and save visualization graphs."""
    sns.set_style("whitegrid")
    agent_df = df[df['role'] == 'AGENT']
    
    y_base, height = 10, 5
    
    # --- 1. TIMELINE ---
    plt.figure(figsize=(15, 6))
    for i, row in df.iterrows():
        color = '#1f77b4' if row['role'] == 'AGENT' else '#2ca02c'
        plt.broken_barh(
            [(row['start'], row['end']-row['start'])],
            (y_base, height),
            facecolors=color,
            edgecolor='white'
        )
    
    for c in cutouts:
        is_silence = "Unusual Silence" in str(c['detected_issues'])
        start, width = c['start'], c['end'] - c['start']
        color = 'red' if is_silence else 'orange'
        
        if is_silence:
            try:
                dur = float([x for x in c['detected_issues'] if "Silence" in x][0].split('(')[1].split('s')[0])
                start = c['end']
                width = dur
            except:
                pass
        
        plt.axvspan(start, start + width, color=color, alpha=0.3, hatch='//')
    
    plt.ylim(0, 25)
    plt.yticks([])
    plt.title('Conversation Timeline & Cutouts', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.legend(
        handles=[
            mpatches.Patch(color='#1f77b4', label='Agent'),
            mpatches.Patch(color='#2ca02c', label='Human'),
            mpatches.Patch(color='red', alpha=0.3, hatch='//', label='Cutout')
        ],
        loc='upper right'
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_graph_1_timeline.png", dpi=150)
    plt.close()
    
    # --- 2. LATENCY BAR ---
    plt.figure(figsize=(10, 6))
    colors = ['red' if x == agent_df['latency'].max() else '#1f77b4' for x in agent_df['latency']]
    sns.barplot(x=agent_df['start'], y=agent_df['latency'], palette=colors)
    plt.title('Agent Response Latency (Over Time)', fontsize=14, fontweight='bold')
    plt.ylabel('Latency (seconds)')
    plt.xlabel('Turn Start Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_graph_2_latency.png", dpi=150)
    plt.close()
    
    # --- 3. DISTRIBUTION ---
    plt.figure(figsize=(8, 6))
    sns.histplot(agent_df['latency'], kde=True, color='purple', bins=10)
    plt.axvline(stats['latency_avg'], color='blue', linestyle='--', label=f"Avg: {stats['latency_avg']:.2f}s")
    plt.axvline(stats['latency_p95'], color='red', linestyle='--', label=f"95%: {stats['latency_p95']:.2f}s")
    plt.title('Latency Distribution Histogram', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_graph_3_distribution.png", dpi=150)
    plt.close()
    
    # --- 4. DASHBOARD ---
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, :])
    for i, row in df.iterrows():
        color = '#1f77b4' if row['role'] == 'AGENT' else '#2ca02c'
        ax1.broken_barh(
            [(row['start'], row['end']-row['start'])],
            (y_base, height),
            facecolors=color,
            edgecolor='white'
        )
    
    for c in cutouts:
        is_silence = "Unusual Silence" in str(c['detected_issues'])
        start, width = c['start'], c['end'] - c['start']
        color = 'red' if is_silence else 'orange'
        
        if is_silence:
            try:
                dur = float([x for x in c['detected_issues'] if "Silence" in x][0].split('(')[1].split('s')[0])
                start = c['end']
                width = dur
            except:
                pass
        
        ax1.axvspan(start, start + width, color=color, alpha=0.3, hatch='//')
    
    ax1.set_title('1. Timeline', fontsize=14)
    ax1.set_yticks([])
    
    ax2 = fig.add_subplot(gs[1, 0])
    sns.barplot(x=agent_df['start'], y=agent_df['latency'], palette=colors, ax=ax2)
    ax2.set_title('2. Latency', fontsize=14)
    ax2.set_xticklabels([])
    
    ax3 = fig.add_subplot(gs[1, 1])
    sns.histplot(agent_df['latency'], kde=True, ax=ax3, color='purple', bins=10)
    ax3.set_title('3. Distribution', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_analysis_dashboard.png", dpi=150)
    plt.close()
    
    print("✅ All graphs saved")


def analyze_transcription(json_file_path):
    """
    Analyze transcription file for cutouts and generate reports.
    
    Args:
        json_file_path: Path to the transcription JSON file
        
    Returns:
        dict: Analysis results with statistics and cutouts
    """
    json_path = Path(json_file_path)
    output_dir = json_path.parent
    base_name = json_path.stem.replace('_original', '').replace('_english', '')
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {json_path.name}")
    print(f"{'='*60}")
    
    # Load models
    load_models()
    
    # Load transcription
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {json_file_path}")
        return None
    
    # 1. PREPROCESS
    raw_segments = data.get('segments', [])
    merged = merge_turns(raw_segments)
    df = pd.DataFrame(merged)
    df['text'] = df['text'].str.strip()
    
    # 2. CALCULATE TIMING
    df['prev_end'] = df['end'].shift(1)
    df['latency'] = df['start'] - df['prev_end']
    df['next_start'] = df['start'].shift(-1)
    df['silence_gap'] = df['next_start'] - df['end']
    
    agent_df = df[df['role'] == 'AGENT'].copy()
    agent_df['latency'] = agent_df['latency'].clip(lower=0).fillna(0)
    agent_df['silence_gap'] = agent_df['silence_gap'].fillna(0)
    
    print(f"🔍 Analyzing {len(agent_df)} Agent turns...")
    
    # 3. FEATURE EXTRACTION
    agent_df['fluency'] = agent_df['text'].apply(calculate_fluency)
    embeddings = _embed_model.encode(agent_df['text'].tolist())
    agent_df['semantic_mag'] = np.linalg.norm(embeddings, axis=1)
    agent_df['len'] = agent_df['text'].apply(len)
    
    # 4. DETECT CUTOUTS WITH CONFIDENCE LEVELS
    agent_df['is_silence'] = get_outliers(agent_df['silence_gap'], config.Z_SCORE_SILENCE)
    agent_df['is_grammar'] = get_outliers(agent_df['fluency'], config.Z_SCORE_GRAMMAR)
    
    len_thresh = agent_df['len'].quantile(config.ACK_PERCENTILE)
    sem_thresh_df = agent_df[agent_df['len'] <= len_thresh]
    sem_thresh = sem_thresh_df['semantic_mag'].quantile(0.25) if len(sem_thresh_df) > 0 else 0
    agent_df['is_ack'] = (agent_df['len'] <= len_thresh) & (agent_df['semantic_mag'] <= sem_thresh)
    
    # Calculate grammar strength (for confidence levels)
    if len(agent_df) > 0:
        grammar_median = agent_df['fluency'].median()
        agent_df['grammar_severity'] = agent_df['fluency'] - grammar_median
    else:
        agent_df['grammar_severity'] = 0
    
    # 5. AGGREGATE RESULTS WITH CONFIDENCE
    cutouts = []
    csv_rows = []
    cutout_durations = []
    
    for idx, row in agent_df.iterrows():
        issues = []
        dur = 0
        confidence = None  # 'definite', 'likely', 'maybe'
        
        # Check if next turn is also AGENT (silence detection only after agent speaks)
        next_speaker = df.loc[df['start'] > row['end']].iloc[0]['role'] if len(df[df['start'] > row['end']]) > 0 else None
        
        # Only flag silence if the NEXT speaker is HUMAN (i.e., agent spoke, then long gap before human)
        # Don't flag if next speaker is AGENT (human took time to respond - that's normal)
        if row['is_silence'] and next_speaker == 'HUMAN':
            issues.append(f"Unusual Silence ({row['silence_gap']:.2f}s)")
            dur += row['silence_gap']
            confidence = 'definite'  # Silence is clear-cut
        
        if row['is_grammar']:
            # Determine confidence based on how far from median
            grammar_z = row['grammar_severity'] / (agent_df['fluency'].std() + 0.001)
            
            if grammar_z > 3.5:  # Very high grammar score
                issues.append(f"Grammar Issue - High (Score {row['fluency']:.2f})")
                confidence = 'definite'
            elif grammar_z > 2.5:  # Moderate grammar score
                issues.append(f"Grammar Issue - Medium (Score {row['fluency']:.2f})")
                confidence = 'likely' if confidence != 'definite' else confidence
            else:  # Low grammar score
                issues.append(f"Grammar Issue - Low (Score {row['fluency']:.2f})")
                confidence = 'maybe' if confidence is None else confidence
            
            dur += (row['end'] - row['start'])
        
        if row['is_ack']:
            issues.append("Empty Acknowledgment")
            dur += (row['end'] - row['start'])
            confidence = 'likely' if confidence is None else confidence
        
        if issues:
            cutouts.append({
                "start": row['start'],
                "end": row['end'],
                "text": row['text'],
                "detected_issues": issues,
                "confidence": confidence if confidence else 'maybe'
            })
            cutout_durations.append(dur)
        
        csv_rows.append({
            "start_time": row['start'],
            "end_time": row['end'],
            "speaker": "AGENT",
            "text": row['text'],
            "latency_seconds": round(row['latency'], 3),
            "is_cutout": bool(issues),
            "cutout_confidence": confidence if issues else "",
            "cutout_details": " + ".join(issues) if issues else ""
        })
    
    # 6. STATISTICS WITH CONFIDENCE BREAKDOWN
    confidence_counts = {
        'definite': len([c for c in cutouts if c['confidence'] == 'definite']),
        'likely': len([c for c in cutouts if c['confidence'] == 'likely']),
        'maybe': len([c for c in cutouts if c['confidence'] == 'maybe'])
    }
    
    stats = {
        "total_turns": len(agent_df),
        "total_cutouts": len(cutouts),
        "cutouts_by_confidence": confidence_counts,
        "latency_avg": float(agent_df['latency'].mean()),
        "latency_max": float(agent_df['latency'].max()),
        "latency_p95": float(agent_df['latency'].quantile(0.95)),
        "cutout_duration_total": sum(cutout_durations)
    }
    
    # 7. EXPORT
    print("="*60)
    print(f"📊 RESULTS:")
    print(f"   Turns: {stats['total_turns']}")
    print(f"   Total Cutouts: {stats['total_cutouts']}")
    print(f"   - Definite: {confidence_counts['definite']}")
    print(f"   - Likely: {confidence_counts['likely']}")
    print(f"   - Maybe: {confidence_counts['maybe']}")
    print(f"   Avg Latency: {stats['latency_avg']:.2f}s")
    print(f"   Total Cutout Duration: {stats['cutout_duration_total']:.2f}s")
    print("="*60)
    
    # Save CSV
    csv_path = output_dir / f"{base_name}_results.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"💾 CSV saved: {csv_path}")
    
    # Save JSON
    json_report_path = output_dir / f"{base_name}_report.json"
    with open(json_report_path, "w") as f:
        json.dump({"summary": stats, "cutouts": cutouts}, f, indent=2)
    print(f"💾 JSON report saved: {json_report_path}")
    
    # Save Graphs
    save_graphs(df, cutouts, stats, output_dir, base_name)
    
    print(f"\n✅ Analysis complete for {json_path.name}\n")
    
    return {"summary": stats, "cutouts": cutouts}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m cutout_pipeline.analyze <transcription_json_path>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    if not Path(json_path).exists():
        print(f"❌ File not found: {json_path}")
        sys.exit(1)
    
    try:
        analyze_transcription(json_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
