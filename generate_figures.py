"""
Generate all draft figures for the Bioalignment manuscript.
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


# === FIGURE 1: Valence-Certainty Quadrant Conceptual Diagram ===

def figure1():
    fig, ax = plt.subplots(figsize=(7, 6))

    # Quadrant colors (light fills)
    # Top-left: Anti-bio/Uncertain
    # Top-right: Pro-bio/Uncertain (IDEAL)
    # Bottom-left: Anti-bio/Certain (WORST)
    # Bottom-right: Pro-bio/Certain

    colors = {
        'anti_uncertain': '#FFF3E0',    # light orange
        'pro_uncertain':  '#E8F5E9',    # light green (ideal)
        'anti_certain':   '#FFEBEE',    # light red (worst)
        'pro_certain':    '#E3F2FD',    # light blue
        'neutral_band':   '#F5F5F5',    # light gray
    }

    # Draw quadrant fills
    # x-axis: valence (Δp_up), y-axis: certainty (σ) — but inverted so "certain" is bottom
    # Actually let's use σ on y-axis with higher = more uncertain (top)

    # Quadrant boundaries
    v_lo, v_hi = -0.25, 0.25
    s_lo, s_hi = 0.0, 0.25

    # Neutral band: -0.05 to +0.05
    neutral_lo, neutral_hi = -0.05, +0.05
    # Certainty threshold
    certain_thresh = 0.10
    moderate_thresh = 0.15

    # Fill quadrants
    # Anti-bio / Certain (bottom-left) — worst
    ax.fill_between([v_lo, neutral_lo], s_lo, certain_thresh,
                    color=colors['anti_certain'], alpha=0.7, zorder=0)
    # Anti-bio / Moderate
    ax.fill_between([v_lo, neutral_lo], certain_thresh, moderate_thresh,
                    color='#FFF8E1', alpha=0.5, zorder=0)
    # Anti-bio / Uncertain (top-left)
    ax.fill_between([v_lo, neutral_lo], moderate_thresh, s_hi,
                    color=colors['anti_uncertain'], alpha=0.7, zorder=0)

    # Pro-bio / Certain (bottom-right)
    ax.fill_between([neutral_hi, v_hi], s_lo, certain_thresh,
                    color=colors['pro_certain'], alpha=0.7, zorder=0)
    # Pro-bio / Moderate
    ax.fill_between([neutral_hi, v_hi], certain_thresh, moderate_thresh,
                    color='#E0F7FA', alpha=0.5, zorder=0)
    # Pro-bio / Uncertain (top-right) — ideal
    ax.fill_between([neutral_hi, v_hi], moderate_thresh, s_hi,
                    color=colors['pro_uncertain'], alpha=0.7, zorder=0)

    # Neutral band (all certainty levels)
    ax.fill_between([neutral_lo, neutral_hi], s_lo, s_hi,
                    color=colors['neutral_band'], alpha=0.5, zorder=0)

    # Draw threshold lines
    ax.axvline(x=neutral_lo, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axvline(x=neutral_hi, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(y=certain_thresh, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(y=moderate_thresh, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    # Quadrant labels
    label_props = dict(fontsize=11, fontweight='bold', ha='center', va='center', alpha=0.8)

    ax.text(-0.15, 0.04, 'Anti-bio\nCertain', **label_props, color='#C62828')
    ax.text(+0.15, 0.04, 'Pro-bio\nCertain', **label_props, color='#1565C0')
    ax.text(-0.15, 0.21, 'Anti-bio\nUncertain', **label_props, color='#E65100')
    ax.text(+0.15, 0.21, 'Pro-bio\nUncertain', **label_props, color='#2E7D32')

    # "WORST" and "IDEAL" annotations
    ax.annotate('WORST', xy=(-0.15, 0.04), xytext=(-0.15, 0.005),
                fontsize=9, fontweight='bold', color='#C62828', ha='center',
                alpha=0.6)
    ax.annotate('IDEAL\n(Bioaligned)', xy=(0.15, 0.21), xytext=(0.15, 0.235),
                fontsize=9, fontweight='bold', color='#2E7D32', ha='center',
                alpha=0.8)

    # Certainty zone labels on right edge
    ax.text(v_hi + 0.005, 0.05, 'Certain\n(σ < 0.10)', fontsize=8, va='center',
            ha='left', color='gray', style='italic')
    ax.text(v_hi + 0.005, 0.125, 'Moderate', fontsize=8, va='center',
            ha='left', color='gray', style='italic')
    ax.text(v_hi + 0.005, 0.20, 'Uncertain\n(σ > 0.15)', fontsize=8, va='center',
            ha='left', color='gray', style='italic')

    # Neutral label
    ax.text(0.0, s_hi + 0.005, 'Neutral\n(±0.05)', fontsize=8, ha='center',
            va='bottom', color='gray', style='italic')

    # Formulas box — centered in the empty middle of the plot
    formula_text = (
        'Valence:  $\\Delta p_{up} = \\overline{p_{up}^{bio}} - \\overline{p_{up}^{nonbio}}$\n'
        'Certainty:  $\\sigma(\\Delta p_{up})$\n'
        'Downside:  $\\Delta p_{down} = \\overline{p_{down}^{nonbio}} - \\overline{p_{down}^{bio}}$'
    )
    ax.text(0.0, 0.125, formula_text, fontsize=9,
            va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.95), zorder=2)

    # Axes
    ax.set_xlabel('Valence ($\\Delta p_{up}$)\n← Anti-biological    Pro-biological →', fontsize=12)
    ax.set_ylabel('Certainty ($\\sigma(\\Delta p_{up})$)\n← Certain    Uncertain →', fontsize=12)
    ax.set_xlim(v_lo, v_hi)
    ax.set_ylim(s_lo, s_hi)
    ax.set_title('The Valence–Certainty Framework', fontsize=14, fontweight='bold', pad=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIGURES_DIR, 'figure1_valence_certainty_framework.png')
    fig.savefig(path)
    path_pdf = os.path.join(FIGURES_DIR, 'figure1_valence_certainty_framework.pdf')
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f'Figure 1 saved: {path}')


# === FIGURE 2: Baseline Models Scatter Plot ===

def figure2():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Quadrant background shading (same scheme as Figure 1)
    v_lo, v_hi = -0.22, 0.10
    s_lo, s_hi = 0.0, 0.20

    neutral_lo, neutral_hi = -0.05, +0.05
    certain_thresh = 0.10
    moderate_thresh = 0.15

    # Light fills
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

    # Plot baseline models
    marker_colors = {
        'Gemma 7B':   '#4CAF50',
        'Llama 8B':   '#2196F3',
        'Phi-3 3.8B': '#9C27B0',
        'Qwen 7B':    '#FF9800',
        'Qwen 3B':    '#F44336',
        'Llama 3B':   '#1976D2',
    }
    markers = {
        'Gemma 7B':   'D',
        'Llama 8B':   's',
        'Phi-3 3.8B': '^',
        'Qwen 7B':    'p',
        'Qwen 3B':    'v',
        'Llama 3B':   'o',
    }

    # Offsets for label placement to avoid overlap
    label_offsets = {
        'Gemma 7B':   (0.006, -0.008),
        'Llama 8B':   (0.006, -0.008),
        'Phi-3 3.8B': (0.006, 0.004),
        'Qwen 7B':    (0.006, 0.004),
        'Qwen 3B':    (0.006, -0.008),
        'Llama 3B':   (0.006, 0.005),
    }

    for name, dp, sigma in baselines:
        ax.scatter(dp, sigma, c=marker_colors[name], marker=markers[name],
                   s=120, zorder=5, edgecolors='black', linewidth=0.8)
        ox, oy = label_offsets[name]
        ax.annotate(name, (dp, sigma), xytext=(dp + ox, sigma + oy),
                    fontsize=9, fontweight='bold', color=marker_colors[name],
                    zorder=6)

    # Plot Llama bioaligned model with different style
    ax.scatter(llama_bioaligned[1], llama_bioaligned[2], c='#4CAF50', marker='*',
               s=250, zorder=5, edgecolors='black', linewidth=0.8)
    ax.annotate('Llama 3B\n(bioaligned)', (llama_bioaligned[1], llama_bioaligned[2]),
                xytext=(llama_bioaligned[1] + 0.008, llama_bioaligned[2] + 0.004),
                fontsize=9, fontweight='bold', color='#4CAF50', zorder=6)

    # Arrow from Llama base to bioaligned
    ax.annotate('', xy=(llama_bioaligned[1], llama_bioaligned[2]),
                xytext=(baselines[-1][1], baselines[-1][2]),
                arrowprops=dict(arrowstyle='->', color='#4CAF50',
                                lw=2, linestyle='--'))

    # Plot Qwen bioaligned model
    ax.scatter(qwen_bioaligned[1], qwen_bioaligned[2], c='#FF5722', marker='*',
               s=250, zorder=5, edgecolors='black', linewidth=0.8)
    ax.annotate('Qwen 3B\n(bioaligned)', (qwen_bioaligned[1], qwen_bioaligned[2]),
                xytext=(qwen_bioaligned[1] + 0.008, qwen_bioaligned[2] - 0.015),
                fontsize=9, fontweight='bold', color='#FF5722', zorder=6)

    # Arrow from Qwen base to bioaligned (Qwen 3B is at index 4 in baselines)
    qwen_base = baselines[4]  # ('Qwen 3B', -0.111, 0.068)
    ax.annotate('', xy=(qwen_bioaligned[1], qwen_bioaligned[2]),
                xytext=(qwen_base[1], qwen_base[2]),
                arrowprops=dict(arrowstyle='->', color='#FF5722',
                                lw=2, linestyle='--'))

    # Quadrant labels (subtle)
    ax.text(-0.175, 0.025, 'Anti-bio / Certain', fontsize=8, color='#B71C1C',
            alpha=0.5, style='italic')
    ax.text(-0.175, 0.125, 'Anti-bio / Moderate', fontsize=8, color='#E65100',
            alpha=0.5, style='italic')
    ax.text(0.06, 0.025, 'Pro-bio\nCertain', fontsize=8, color='#0D47A1',
            alpha=0.5, style='italic')

    ax.set_xlabel('Valence ($\\Delta p_{up}$)\n← Anti-biological    Pro-biological →', fontsize=12)
    ax.set_ylabel('Certainty ($\\sigma(\\Delta p_{up})$)\n← Certain    Uncertain →', fontsize=12)
    ax.set_xlim(v_lo, v_hi)
    ax.set_ylim(s_lo, s_hi)
    ax.set_title('Baseline Model Dispositions Toward Biology', fontsize=14,
                 fontweight='bold', pad=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIGURES_DIR, 'figure2_baseline_scatter.png')
    fig.savefig(path)
    path_pdf = os.path.join(FIGURES_DIR, 'figure2_baseline_scatter.pdf')
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f'Figure 2 saved: {path}')


# === FIGURE 3: Before/After Bar Chart (Llama and Qwen) ===

def figure3():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Four bars: Llama base, Llama bio, Qwen base, Qwen bio
    models = ['Llama 3B\nBase', 'Llama 3B\nBioaligned', 'Qwen 3B\nBase', 'Qwen 3B\nBioaligned']
    values = [main_base_dp, main_bio_dp, qwen_base_dp, qwen_bio_dp]

    # Error bars from 95% CI
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

    # Improvement annotations
    ax.annotate('93%', xy=(0.5, -0.075), fontsize=14, fontweight='bold',
                ha='center', color='#2E7D32',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9',
                          edgecolor='#4CAF50', alpha=0.9))
    ax.annotate('51%', xy=(2.5, -0.085), fontsize=14, fontweight='bold',
                ha='center', color='#E65100',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF3E0',
                          edgecolor='#FF9800', alpha=0.9))

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        y_pos = val - 0.008 if val < -0.02 else val + 0.008
        va = 'top' if val < -0.02 else 'bottom'
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'{val:.3f}', ha='center', va=va, fontsize=10, fontweight='bold',
                color='white' if val < -0.05 else 'black')

    ax.set_ylabel('$\\Delta p_{up}$ (Valence)', fontsize=13)
    ax.set_ylim(-0.22, 0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_title('Bias Reduction After QLoRA Fine-Tuning', fontsize=14,
                 fontweight='bold', pad=12)

    # Annotation for direction
    ax.text(-0.6, -0.19, '← Anti-biological', fontsize=9, color='gray',
            style='italic', ha='left')
    ax.text(-0.6, 0.09, '← Pro-biological', fontsize=9, color='gray',
            style='italic', ha='left')

    # Group labels
    ax.text(0.5, -0.205, 'Llama 3.2 3B Instruct', ha='center', fontsize=9,
            fontweight='bold', color='#1976D2')
    ax.text(2.5, -0.205, 'Qwen 2.5 3B Instruct', ha='center', fontsize=9,
            fontweight='bold', color='#E65100')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, 'figure3_before_after.png')
    fig.savefig(path)
    path_pdf = os.path.join(FIGURES_DIR, 'figure3_before_after.pdf')
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f'Figure 3 saved: {path}')


# === FIGURE 4: Training Dynamics (Two-Panel: Llama and Qwen) ===

def figure4():
    """Two-panel figure showing training dynamics for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ========== PANEL A: Llama 3B Δp_up trajectory ==========
    steps = [c[0] for c in llama_checkpoints]
    dp_values = [c[1] for c in llama_checkpoints]

    ax1.plot(steps, dp_values, 'o-', color='#1976D2', linewidth=2,
            markersize=7, markeredgecolor='black', markeredgewidth=0.8,
            markerfacecolor='#1976D2', zorder=5)

    ax1.scatter([0], [-0.141], c='#E53935', s=150, marker='s', zorder=6,
               edgecolors='black', linewidth=1.2, label='Base model')
    ax1.scatter([800], [-0.002], c='#4CAF50', s=150, marker='*', zorder=6,
               edgecolors='black', linewidth=1.2, label='Selected checkpoint')

    ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
    ax1.fill_between([min(steps) - 20, max(steps) + 20], -0.05, +0.05,
                    color='#E8F5E9', alpha=0.3, label='Neutral zone (±0.05)')
    ax1.axhline(y=-0.05, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax1.axhline(y=+0.05, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('$\\Delta p_{up}$ (Valence)', fontsize=12)
    ax1.set_title('(A) Llama 3B: Bioalignment Trajectory', fontsize=13,
                 fontweight='bold', pad=10)
    ax1.set_xlim(-30, 1150)
    ax1.set_ylim(-0.18, 0.12)
    ax1.legend(loc='lower right', framealpha=0.9, edgecolor='gray', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.text(0.02, 0.98, 'Result: 93% improvement\n$\\Delta p_{up}$: -0.141 → -0.009',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                      edgecolor='#4CAF50', alpha=0.9))

    # ========== PANEL B: Qwen 3B Loss Curve ==========
    qwen_steps = [d[0] for d in qwen_loss_data]
    qwen_loss = [d[1] for d in qwen_loss_data]

    ax2.plot(qwen_steps, qwen_loss, 'o-', color='#F44336', linewidth=2,
            markersize=7, markeredgecolor='black', markeredgewidth=0.8,
            markerfacecolor='#F44336', zorder=5)

    ax2.scatter([0], [9.09], c='#E53935', s=150, marker='s', zorder=6,
               edgecolors='black', linewidth=1.2, label='Initial loss')
    ax2.scatter([408], [7.95], c='#4CAF50', s=150, marker='*', zorder=6,
               edgecolors='black', linewidth=1.2, label='Final loss')

    epoch_boundaries = [136, 272]
    for i, boundary in enumerate(epoch_boundaries):
        ax2.axvline(x=boundary, color='gray', linewidth=1.2, linestyle='--', alpha=0.6)
        ax2.text(boundary + 5, 9.0, f'Epoch {i+2}', fontsize=8, color='gray',
                rotation=0, va='bottom', style='italic')

    ax2.text(5, 9.0, 'Epoch 1', fontsize=8, color='gray', va='bottom', style='italic')

    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('(B) Qwen 3B: Training Loss Curve', fontsize=13,
                 fontweight='bold', pad=10)
    ax2.set_xlim(-10, 430)
    ax2.set_ylim(7.5, 9.5)
    ax2.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.text(0.02, 0.98, 'Result: 51% improvement\n$\\Delta p_{up}$: -0.111 → -0.056',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                      edgecolor='#F44336', alpha=0.9))

    ax2.text(0.98, 0.02, 'lr=1e-5, 544 examples\n3 epochs, LoRA r=16',
             transform=ax2.transAxes, fontsize=8, verticalalignment='bottom',
             ha='right', color='gray', style='italic')

    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, 'figure4_training_dynamics.png')
    fig.savefig(path)
    path_pdf = os.path.join(FIGURES_DIR, 'figure4_training_dynamics.pdf')
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f'Figure 4 saved: {path}')


# === Generate all figures ===

if __name__ == '__main__':
    print('Generating figures...')
    figure1()
    figure2()
    figure3()
    figure4()
    print(f'\nAll figures saved to: {FIGURES_DIR}')
