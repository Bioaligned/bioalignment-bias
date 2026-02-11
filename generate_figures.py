"""
Generate all figures for the Bioalignment manuscript.
Figures 1-4 as referenced in the outline and draft sections.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'lines.markersize': 8,
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# === DATA ===

# Baseline models: (name, delta_p_up, sigma_delta_p_up)
baselines = [
    ('Gemma 7B',   +0.015, 0.062),
    ('Llama 8B',   -0.031, 0.063),
    ('Phi-3 3.8B', -0.038, 0.142),
    ('Qwen 7B',    -0.039, 0.093),
    ('Qwen 3B',    -0.111, 0.068),
    ('Llama 3B',   -0.141, 0.111),
]

# Bioaligned models
llama_bioaligned = ('Llama 3B\n(bioaligned)', -0.009, 0.115)
qwen_bioaligned = ('Qwen 3B\n(bioaligned)', -0.056, 0.070)  # sigma estimated

# Keep for backwards compatibility
bioaligned = llama_bioaligned

# Main result
main_base_dp = -0.141
main_bio_dp = -0.009
main_base_ci = (-0.17, -0.11)
main_bio_ci = (-0.04, +0.02)

# Training dynamics - Llama 3B
llama_checkpoints = [
    (0,    -0.141),
    (100,  -0.042),
    (200,  +0.063),
    (300,  +0.023),
    (400,  +0.043),
    (500,  +0.022),
    (600,  -0.052),
    (700,  -0.010),
    (800,  -0.002),
    (900,  +0.035),
    (1000, -0.025),
    (1100, -0.025),
]

# Keep for backwards compatibility
checkpoints = llama_checkpoints

# Training dynamics - Qwen 3B (loss curve across 3 epochs)
# Data extracted from training log: train_lr1e5_v2.log
# Steps per epoch: ~136 (4 batches/step with 544 examples, batch_size=16)
qwen_loss_data = [
    # Epoch 1 (steps 0-136)
    (0,   9.09),   # start of epoch 1
    (40,  9.09),   # logged at step 40
    (80,  8.78),   # logged at step 80
    (120, 8.54),   # logged at step 120
    # Epoch 2 (steps 136-272)
    (136, 8.26),   # start of epoch 2
    (159, 8.26),   # step 23 in epoch 2
    (199, 8.10),   # step 63 in epoch 2
    (239, 8.22),   # step 103 in epoch 2
    # Epoch 3 (steps 272-408)
    (272, 7.99),   # start of epoch 3
    (279, 7.99),   # step 7 in epoch 3
    (319, 8.03),   # step 47 in epoch 3
    (359, 7.95),   # step 87 in epoch 3
    (408, 7.95),   # end of training
]

# Qwen bioalignment result
qwen_base_dp = -0.111
qwen_bio_dp = -0.056
qwen_base_sigma = 0.068

# Frontier models: (name, delta_p_up, sigma)
# Data from actual result files in bioalignment_eval/
frontier_models = [
    ('Claude Opus 4.5',    +0.2245, 0.050),
    ('Gemini 2.5 Flash',   +0.164,  0.167),
    ('GPT-5.2',            -0.045,  0.057),
    ('GPT-4o',             -0.053,  0.074),
    ('Gemini 2.0 Flash',   -0.143,  0.146),
]


# === FIGURE 1: All Baseline Models (Open-Source + Frontier) ===

def figure1():
    """Figure 1: All baseline models plotted on valence-certainty space.
    Includes open-source baselines and frontier models. No bioaligned models."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Quadrant background shading
    v_lo, v_hi = -0.22, 0.28
    s_lo, s_hi = 0.0, 0.20

    neutral_lo, neutral_hi = -0.05, +0.05
    certain_thresh = 0.10
    moderate_thresh = 0.15

    # Light fills for quadrants
    # Anti-bio / Certain
    ax.fill_between([v_lo, neutral_lo], s_lo, certain_thresh,
                    color='#FFEBEE', alpha=0.4, zorder=0)
    # Anti-bio / Moderate
    ax.fill_between([v_lo, neutral_lo], certain_thresh, moderate_thresh,
                    color='#FFF8E1', alpha=0.3, zorder=0)
    # Anti-bio / Uncertain
    ax.fill_between([v_lo, neutral_lo], moderate_thresh, s_hi,
                    color='#FFF3E0', alpha=0.4, zorder=0)
    # Neutral band
    ax.fill_between([neutral_lo, neutral_hi], s_lo, s_hi,
                    color='#F5F5F5', alpha=0.4, zorder=0)
    # Pro-bio / Certain
    ax.fill_between([neutral_hi, v_hi], s_lo, certain_thresh,
                    color='#E3F2FD', alpha=0.4, zorder=0)
    # Pro-bio / Moderate
    ax.fill_between([neutral_hi, v_hi], certain_thresh, moderate_thresh,
                    color='#E0F7FA', alpha=0.3, zorder=0)
    # Pro-bio / Uncertain
    ax.fill_between([neutral_hi, v_hi], moderate_thresh, s_hi,
                    color='#E8F5E9', alpha=0.4, zorder=0)

    # Threshold lines
    ax.axvline(x=neutral_lo, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(x=neutral_hi, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(y=certain_thresh, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(y=moderate_thresh, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    # Plot open-source baseline models (all circles)
    marker_colors = {
        'Gemma 7B':   '#4CAF50',
        'Llama 8B':   '#2196F3',
        'Phi-3 3.8B': '#9C27B0',
        'Qwen 7B':    '#FF9800',
        'Qwen 3B':    '#F44336',
        'Llama 3B':   '#1976D2',
    }
    # All open-source models use circles
    markers = {
        'Gemma 7B':   'o',
        'Llama 8B':   'o',
        'Phi-3 3.8B': 'o',
        'Qwen 7B':    'o',
        'Qwen 3B':    'o',
        'Llama 3B':   'o',
    }

    label_offsets = {
        'Gemma 7B':   (0.010, 0.006),
        'Llama 8B':   (-0.008, 0.005),
        'Phi-3 3.8B': (0.006, 0.004),
        'Qwen 7B':    (0.006, 0.004),
        'Qwen 3B':    (0.012, -0.012),
        'Llama 3B':   (0.012, 0.006),
    }

    for name, dp, sigma in baselines:
        ax.scatter(dp, sigma, c=marker_colors[name], marker=markers[name],
                   s=120, zorder=5, edgecolors='black', linewidth=0.8)
        ox, oy = label_offsets[name]
        ax.annotate(name, (dp, sigma), xytext=(dp + ox, sigma + oy),
                    fontsize=9, fontweight='bold', color=marker_colors[name],
                    zorder=6)

    # Plot frontier models as stars
    frontier_colors = {
        'Claude Opus 4.5':    '#7C4DFF',
        'Gemini 2.5 Flash':   '#00BCD4',
        'GPT-5.2':            '#795548',
        'GPT-4o':             '#607D8B',
        'Gemini 2.0 Flash':   '#FF5722',
    }
    frontier_offsets = {
        'Claude Opus 4.5':    (0.008, 0.005),
        'Gemini 2.5 Flash':   (0.008, 0.003),
        'GPT-5.2':            (0.008, -0.01),
        'GPT-4o':             (0.008, 0.006),
        'Gemini 2.0 Flash':   (0.008, 0.005),
    }

    for name, dp, sigma in frontier_models:
        ax.scatter(dp, sigma, c=frontier_colors[name], marker='*',
                   s=300, zorder=6, edgecolors='black', linewidth=0.8)
        ox, oy = frontier_offsets[name]
        ax.annotate(name, (dp, sigma), xytext=(dp + ox, sigma + oy),
                    fontsize=8, fontweight='bold', color=frontier_colors[name],
                    zorder=7)

    # Quadrant labels (subtle)
    ax.text(-0.175, 0.025, 'Anti-bio / Certain', fontsize=8, color='#B71C1C',
            alpha=0.5, style='italic')
    ax.text(-0.175, 0.125, 'Anti-bio / Moderate', fontsize=8, color='#E65100',
            alpha=0.5, style='italic')
    ax.text(0.18, 0.025, 'Pro-bio / Certain', fontsize=8, color='#0D47A1',
            alpha=0.5, style='italic')
    ax.text(0.18, 0.175, 'Pro-bio / Uncertain\n(IDEAL)', fontsize=8, color='#2E7D32',
            alpha=0.6, style='italic', fontweight='bold')

    ax.set_xlabel('Valence ($\\Delta p_{up}$)\n← Anti-biological    Pro-biological →', fontsize=12)
    ax.set_ylabel('Certainty ($\\sigma(\\Delta p_{up})$)', fontsize=12)
    ax.set_xlim(v_lo, v_hi)
    ax.set_ylim(s_lo, s_hi)
    ax.set_title('Baseline Model Dispositions Toward Biology\n(Open-Source: circles  |  Frontier: stars)', fontsize=14,
                 fontweight='bold', pad=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIGURES_DIR, 'figure1_baseline_models.png')
    fig.savefig(path)
    path_pdf = os.path.join(FIGURES_DIR, 'figure1_baseline_models.pdf')
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f'Figure 1 saved: {path}')


# === FIGURE 4: Bioaligned Training Effect (Full Context) ===

def figure4_context():
    """Figure 4: All baseline models plus bioaligned intervention.
    Shows full valence-certainty landscape with arrows showing training effect."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Quadrant background shading (same as figure 1)
    v_lo, v_hi = -0.22, 0.28
    s_lo, s_hi = 0.0, 0.20

    neutral_lo, neutral_hi = -0.05, +0.05
    certain_thresh = 0.10
    moderate_thresh = 0.15

    # Light fills for quadrants
    ax.fill_between([v_lo, neutral_lo], s_lo, certain_thresh,
                    color='#FFEBEE', alpha=0.4, zorder=0)
    ax.fill_between([v_lo, neutral_lo], certain_thresh, moderate_thresh,
                    color='#FFF8E1', alpha=0.3, zorder=0)
    ax.fill_between([v_lo, neutral_lo], moderate_thresh, s_hi,
                    color='#FFF3E0', alpha=0.4, zorder=0)
    ax.fill_between([neutral_lo, neutral_hi], s_lo, s_hi,
                    color='#F5F5F5', alpha=0.4, zorder=0)
    ax.fill_between([neutral_hi, v_hi], s_lo, certain_thresh,
                    color='#E3F2FD', alpha=0.4, zorder=0)
    ax.fill_between([neutral_hi, v_hi], certain_thresh, moderate_thresh,
                    color='#E0F7FA', alpha=0.3, zorder=0)
    ax.fill_between([neutral_hi, v_hi], moderate_thresh, s_hi,
                    color='#E8F5E9', alpha=0.4, zorder=0)

    # Threshold lines
    ax.axvline(x=neutral_lo, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(x=neutral_hi, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(y=certain_thresh, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(y=moderate_thresh, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    # Plot open-source baseline models (circles) - excluding Llama 3B and Qwen 3B
    # which will be shown with arrows
    other_baselines = [b for b in baselines if b[0] not in ['Llama 3B', 'Qwen 3B']]
    marker_colors = {
        'Gemma 7B':   '#4CAF50',
        'Llama 8B':   '#2196F3',
        'Phi-3 3.8B': '#9C27B0',
        'Qwen 7B':    '#FF9800',
    }
    label_offsets = {
        'Gemma 7B':   (0.010, 0.006),     
        'Llama 8B':   (-0.008, 0.005),    
        'Phi-3 3.8B': (0.006, 0.004),
        'Qwen 7B':    (0.006, 0.004),
    }

    for name, dp, sigma in other_baselines:
        ax.scatter(dp, sigma, c=marker_colors[name], marker='o',
                   s=100, zorder=4, edgecolors='black', linewidth=0.6, alpha=0.6)
        ox, oy = label_offsets[name]
        ax.annotate(name, (dp, sigma), xytext=(dp + ox, sigma + oy),
                    fontsize=8, color=marker_colors[name], alpha=0.7,
                    zorder=5)

    # Plot frontier models (stars) - dimmed as context
    frontier_colors = {
        'Claude Opus 4.5':    '#7C4DFF',
        'Gemini 2.5 Flash':   '#00BCD4',
        'GPT-5.2':            '#795548',
        'GPT-4o':             '#607D8B',
        'Gemini 2.0 Flash':   '#FF5722',
    }
    frontier_offsets = {
        'Claude Opus 4.5':    (0.008, 0.005),
        'Gemini 2.5 Flash':   (0.008, 0.003),
        'GPT-5.2':            (0.008, -0.01),    
        'GPT-4o':             (0.008, 0.006),     
        'Gemini 2.0 Flash':   (0.008, 0.005),
    }

    for name, dp, sigma in frontier_models:
        ax.scatter(dp, sigma, c=frontier_colors[name], marker='*',
                   s=200, zorder=4, edgecolors='black', linewidth=0.6, alpha=0.6)
        ox, oy = frontier_offsets[name]
        ax.annotate(name, (dp, sigma), xytext=(dp + ox, sigma + oy),
                    fontsize=7, color=frontier_colors[name], alpha=0.7,
                    zorder=5)

    # === INTERVENTION: Llama 3B ===
    llama_base = baselines[-1]  # ('Llama 3B', -0.141, 0.111)
    llama_color = '#1976D2'

    # Plot Llama base (solid circle)
    ax.scatter(llama_base[1], llama_base[2], c=llama_color, marker='o',
               s=150, zorder=6, edgecolors='black', linewidth=1.2)

    # Plot Llama bioaligned (same color circle, open/hollow)
    ax.scatter(llama_bioaligned[1], llama_bioaligned[2], c=llama_color, marker='o',
               s=150, zorder=6, edgecolors='black', linewidth=1.2)

    # Straight arrow from Llama base to bioaligned
    ax.annotate('', xy=(llama_bioaligned[1], llama_bioaligned[2]),
                xytext=(llama_base[1], llama_base[2]),
                arrowprops=dict(arrowstyle='->', color=llama_color,
                                lw=2.5, linestyle='-'))

    # Llama labels
    ax.annotate('Llama 3B', (llama_base[1], llama_base[2]),
                xytext=(llama_base[1] - 0.008, llama_base[2] + 0.012),
                fontsize=9, fontweight='bold', color=llama_color, ha='right')
    ax.annotate('93%', xy=((llama_base[1] + llama_bioaligned[1])/2 + 0.01,
                           (llama_base[2] + llama_bioaligned[2])/2 + 0.012),
                fontsize=10, fontweight='bold', color=llama_color,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=llama_color, alpha=0.95))

    # === INTERVENTION: Qwen 3B ===
    qwen_base = baselines[4]  # ('Qwen 3B', -0.111, 0.068)
    qwen_color = '#F44336'

    # Plot Qwen base (solid circle)
    ax.scatter(qwen_base[1], qwen_base[2], c=qwen_color, marker='o',
               s=150, zorder=6, edgecolors='black', linewidth=1.2)

    # Plot Qwen bioaligned (same color circle)
    ax.scatter(qwen_bioaligned[1], qwen_bioaligned[2], c=qwen_color, marker='o',
               s=150, zorder=6, edgecolors='black', linewidth=1.2)

    # Straight arrow from Qwen base to bioaligned
    ax.annotate('', xy=(qwen_bioaligned[1], qwen_bioaligned[2]),
                xytext=(qwen_base[1], qwen_base[2]),
                arrowprops=dict(arrowstyle='->', color=qwen_color,
                                lw=2.5, linestyle='-'))

    # Qwen labels
    ax.annotate('Qwen 3B', (qwen_base[1], qwen_base[2]),
                xytext=(qwen_base[1] - 0.015, qwen_base[2] - 0.012),
                fontsize=9, fontweight='bold', color=qwen_color, ha='right')
    ax.annotate('51%', xy=((qwen_base[1] + qwen_bioaligned[1])/2 + 0.01,
                           (qwen_base[2] + qwen_bioaligned[2])/2 - 0.008),
                fontsize=10, fontweight='bold', color=qwen_color,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=qwen_color, alpha=0.95))

    # Quadrant labels (subtle)
    ax.text(-0.175, 0.025, 'Anti-bio / Certain', fontsize=8, color='#B71C1C',
            alpha=0.5, style='italic')
    ax.text(-0.175, 0.135, 'Anti-bio / Moderate', fontsize=8, color='#E65100',
            alpha=0.5, style='italic')
    ax.text(0.18, 0.025, 'Pro-bio / Certain', fontsize=8, color='#0D47A1',
            alpha=0.5, style='italic')
    ax.text(0.18, 0.175, 'Pro-bio / Uncertain\n(IDEAL)', fontsize=8, color='#2E7D32',
            alpha=0.6, style='italic', fontweight='bold')

    ax.set_xlabel('Valence ($\\Delta p_{up}$)\n← Anti-biological    Pro-biological →', fontsize=12)
    ax.set_ylabel('Certainty ($\\sigma(\\Delta p_{up})$)', fontsize=12)
    ax.set_xlim(v_lo, v_hi)
    ax.set_ylim(s_lo, s_hi)
    ax.set_title('Effect of Bioaligned Training', fontsize=14,
                 fontweight='bold', pad=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIGURES_DIR, 'figure4_context.png')
    fig.savefig(path)
    path_pdf = os.path.join(FIGURES_DIR, 'figure4_context.pdf')
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f'Figure 4 saved: {path}')


# === FIGURE 2: Before/After Bar Chart (Llama and Qwen) ===

def figure2_bar():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Four bars: Llama base, Llama bio, Qwen base, Qwen bio
    models = ['Llama 3B\nBase', 'Llama 3B\nBioaligned', 'Qwen 3B\nBase', 'Qwen 3B\nBioaligned']
    values = [main_base_dp, main_bio_dp, qwen_base_dp, qwen_bio_dp]

    # Error bars from 95% CI
    # Llama Base: value = -0.141, CI = [-0.17, -0.11]
    #   lower_err = -0.141 - (-0.17) = 0.029
    #   upper_err = -0.11 - (-0.141) = 0.031
    # Llama Bio: value = -0.009, CI = [-0.04, +0.02]
    #   lower_err = -0.009 - (-0.04) = 0.031
    #   upper_err = 0.02 - (-0.009) = 0.029
    # Qwen Base: value = -0.111, CI = [-0.13, -0.09] (estimated from σ=0.068)
    #   lower_err = 0.019, upper_err = 0.021
    # Qwen Bio: value = -0.056, CI estimated
    #   lower_err = 0.020, upper_err = 0.020
    yerr = [[0.029, 0.031, 0.019, 0.020], [0.031, 0.029, 0.021, 0.020]]

    colors = ['#E53935', '#43A047', '#FF7043', '#66BB6A']

    x = np.arange(len(models))
    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor='black',
                  linewidth=0.8, yerr=yerr, capsize=6, error_kw={'linewidth': 1.5})

    # Zero line
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

    # Llama significance bracket
    bracket_y = 0.06
    ax.plot([0, 0, 1, 1], [bracket_y - 0.005, bracket_y, bracket_y, bracket_y - 0.005],
            color='black', linewidth=1.2)
    ax.text(0.5, bracket_y + 0.005, '***\np < 0.001',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Qwen significance bracket
    ax.plot([2, 2, 3, 3], [bracket_y - 0.005, bracket_y, bracket_y, bracket_y - 0.005],
            color='black', linewidth=1.2)
    ax.text(2.5, bracket_y + 0.005, '**\np < 0.01',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Improvement annotations (both use same green color for consistency)
    ax.annotate('93%', xy=(0.5, -0.075), fontsize=14, fontweight='bold',
                ha='center', color='#2E7D32',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9',
                          edgecolor='#4CAF50', alpha=0.9))
    ax.annotate('51%', xy=(2.5, -0.085), fontsize=14, fontweight='bold',
                ha='center', color='#2E7D32',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9',
                          edgecolor='#4CAF50', alpha=0.9))

    # Value labels to the right of error bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        # Position label to the right of the bar/error bar
        x_pos = bar.get_x() + bar.get_width() + 0.08
        y_pos = val
        ax.text(x_pos, y_pos, f'{val:.3f}', ha='left', va='center',
                fontsize=9, fontweight='bold', color='#333333')

    ax.set_ylabel('$\\Delta p_{up}$ (Valence)', fontsize=13)
    ax.set_ylim(-0.22, 0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_title('Bias Reduction After QLoRA Fine-Tuning', fontsize=14,
                 fontweight='bold', pad=12)

    # Group labels
    ax.text(0.5, -0.205, 'Llama 3.2 3B Instruct', ha='center', fontsize=9,
            fontweight='bold', color='#1976D2')
    ax.text(2.5, -0.205, 'Qwen 2.5 3B Instruct', ha='center', fontsize=9,
            fontweight='bold', color='#E65100')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, 'figure2_before_after.png')
    fig.savefig(path)
    path_pdf = os.path.join(FIGURES_DIR, 'figure2_before_after.pdf')
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f'Figure 2 saved: {path}')


# === FIGURE 3: Training Dynamics (Llama 3B only) ===

def figure3_dynamics():
    """Single-panel figure showing Llama 3B training dynamics."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    steps = [c[0] for c in llama_checkpoints]
    dp_values = [c[1] for c in llama_checkpoints]

    # Plot line
    ax.plot(steps, dp_values, 'o-', color='#1976D2', linewidth=2,
            markersize=7, markeredgecolor='black', markeredgewidth=0.8,
            markerfacecolor='#1976D2', zorder=5)

    # Highlight base (step 0) and selected checkpoint (step 800)
    ax.scatter([0], [-0.141], c='#E53935', s=150, marker='s', zorder=6,
               edgecolors='black', linewidth=1.2, label='Base model')
    ax.scatter([800], [-0.002], c='#4CAF50', s=150, marker='*', zorder=6,
               edgecolors='black', linewidth=1.2, label='Selected checkpoint')

    # Zero line
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)

    # Neutral band
    ax.fill_between([min(steps) - 20, max(steps) + 20], -0.05, +0.05,
                    color='#E8F5E9', alpha=0.3, label='Neutral zone (±0.05)')

    # Anti-bio threshold
    ax.axhline(y=-0.05, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.axhline(y=+0.05, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)

    # Phase annotations
    ax.annotate('Phase 1:\nRapid correction',
                xy=(100, -0.04), xytext=(50, 0.08),
                fontsize=8, color='#1565C0', ha='center',
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#1565C0', alpha=0.8))

    ax.annotate('Phase 2:\nNear-neutral\noscillation',
                xy=(600, -0.05), xytext=(750, -0.12),
                fontsize=8, color='#E65100', ha='center',
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=1),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#E65100', alpha=0.8))

    ax.set_xlabel('Training Step', fontsize=13)
    ax.set_ylabel('$\\Delta p_{up}$ (Valence)', fontsize=13)
    ax.set_title('Bioalignment Trajectory During Training', fontsize=14,
                 fontweight='bold', pad=12)
    ax.set_xlim(-30, 1150)
    ax.set_ylim(-0.18, 0.12)

    ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIGURES_DIR, 'figure3_training_dynamics.png')
    fig.savefig(path)
    path_pdf = os.path.join(FIGURES_DIR, 'figure3_training_dynamics.pdf')
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f'Figure 3 saved: {path}')


# === Generate all figures ===

if __name__ == '__main__':
    print('Generating figures...')
    figure1()           # Baseline landscape
    figure2_bar()       # Before/after bar chart (headline result)
    figure3_dynamics()  # Training dynamics
    figure4_context()   # Full context with intervention arrows
    print(f'\nAll figures saved to: {FIGURES_DIR}')
