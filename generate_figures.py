"""
Generate all figures for the Bioalignment manuscript.
Figures 1-4 as referenced in the outline and draft sections.

Updated to use bar charts instead of 2D scatter plots (simplified framework).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


# === DATA (recalculated from JSON_results_all_models) ===

# All baseline models sorted by Bioalignment Metric (delta_p_up)
# Format: (name, delta_p_up, sigma, type)
# type: 'frontier' or 'open-weight'
all_models = [
    ('Claude Opus 4.5',    +0.224, 0.055, 'frontier'),
    ('Gemini 2.5 Flash',   +0.164, 0.166, 'frontier'),
    ('Mistral 7B',         +0.059, 0.111, 'open-weight'),
    ('Llama 8B',           -0.031, 0.064, 'open-weight'),
    ('Phi-3 3.8B',         -0.038, 0.143, 'open-weight'),
    ('GPT-5.2',            -0.045, 0.057, 'frontier'),
    ('GPT-4o',             -0.053, 0.074, 'frontier'),
    ('Qwen 3B',            -0.111, 0.069, 'open-weight'),
    ('Llama 3B',           -0.141, 0.112, 'open-weight'),
    ('Gemini 2.0 Flash',   -0.143, 0.146, 'frontier'),
]

# Before/after data for fine-tuning results
llama_base = ('Llama 3B', -0.141, 0.112)
llama_bioaligned = ('Llama 3B (bioaligned)', -0.009, 0.116)
qwen_base = ('Qwen 3B', -0.111, 0.069)
qwen_bioaligned = ('Qwen 3B (bioaligned)', -0.057, 0.089)

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


# === FIGURE 1: Vertical Bar Chart of All Models ===

def figure1_bar():
    """Figure 1: Vertical bar chart showing all models sorted by Bioalignment Metric.
    Color-coded: red for pro-synthetic (negative), gray for neutral, blue for pro-bio (positive).
    Error bars show +/- 1 sigma. Frontier models shown in bold."""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort models by delta_p_up (most positive on left)
    models_sorted = sorted(all_models, key=lambda x: x[1], reverse=True)

    names = [m[0] for m in models_sorted]
    values = [m[1] for m in models_sorted]
    sigmas = [m[2] for m in models_sorted]
    types = [m[3] for m in models_sorted]

    x_pos = np.arange(len(names))

    # Color based on value: blue for positive, red for negative, gray for neutral
    colors = []
    for v in values:
        if v > 0.05:
            colors.append('#1976D2')  # Blue for pro-bio
        elif v < -0.05:
            colors.append('#D32F2F')  # Red for pro-synthetic
        else:
            colors.append('#757575')  # Gray for neutral

    # Create vertical bars (no error bars - sigma values are in tables)
    bars = ax.bar(x_pos, values, width=0.7,
                  color=colors, edgecolor='black', linewidth=0.8)

    # Zero line
    ax.axhline(y=0, color='black', linewidth=1.0, linestyle='-')

    # Neutral zone shading
    ax.axhspan(-0.05, 0.05, color='#E8E8E8', alpha=0.5, zorder=0)

    # Labels and formatting - bold for frontier models
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11, rotation=45, ha='right')
    for i, (label, mtype) in enumerate(zip(ax.get_xticklabels(), types)):
        if mtype == 'frontier':
            label.set_fontweight('bold')
    ax.set_ylabel('Bioalignment Metric ($\\Delta p_{up}$)', fontsize=13)
    ax.set_ylim(-0.22, 0.32)
    ax.set_title('Model Bioalignment Scores',
                 fontsize=14, fontweight='bold', pad=12)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1976D2', linewidth=8, label='Pro-biological ($>$+0.05)'),
        Line2D([0], [0], color='#757575', linewidth=8, label='Neutral ($\\pm$0.05)'),
        Line2D([0], [0], color='#D32F2F', linewidth=8, label='Pro-synthetic ($<$-0.05)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              framealpha=0.95, edgecolor='gray')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, 'figure1_baseline_models.png')
    fig.savefig(path)
    path_pdf = os.path.join(FIGURES_DIR, 'figure1_baseline_models.pdf')
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f'Figure 1 saved: {path}')


# === FIGURE 2: Before/After Bar Chart (Llama and Qwen) ===

def figure2_bar():
    """Figure 2: Grouped bar chart showing before/after for Llama 3B and Qwen 3B."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Four bars: Llama base, Llama bio, Qwen base, Qwen bio
    models = ['Llama 3B\nBase', 'Llama 3B\nBioaligned', 'Qwen 3B\nBase', 'Qwen 3B\nBioaligned']
    values = [llama_base[1], llama_bioaligned[1], qwen_base[1], qwen_bioaligned[1]]
    sigmas = [llama_base[2], llama_bioaligned[2], qwen_base[2], qwen_bioaligned[2]]

    colors = ['#E53935', '#43A047', '#FF7043', '#66BB6A']

    x = np.arange(len(models))
    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor='black',
                  linewidth=0.8, yerr=sigmas, capsize=6, error_kw={'linewidth': 1.5})

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

    # Shift annotations (absolute change in delta_p_up)
    llama_shift = llama_bioaligned[1] - llama_base[1]
    qwen_shift = qwen_bioaligned[1] - qwen_base[1]

    ax.annotate(f'+{llama_shift:.3f}', xy=(0.5, -0.075), fontsize=14, fontweight='bold',
                ha='center', color='#2E7D32',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9',
                          edgecolor='#4CAF50', alpha=0.9))
    ax.annotate(f'+{qwen_shift:.3f}', xy=(2.5, -0.085), fontsize=14, fontweight='bold',
                ha='center', color='#2E7D32',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9',
                          edgecolor='#4CAF50', alpha=0.9))

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        x_pos = bar.get_x() + bar.get_width() + 0.08
        y_pos = val
        ax.text(x_pos, y_pos, f'{val:.3f}', ha='left', va='center',
                fontsize=9, fontweight='bold', color='#333333')

    ax.set_ylabel('Bioalignment Metric ($\\Delta p_{up}$)', fontsize=13)
    ax.set_ylim(-0.3, 0.12)
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

    # Highlight base (step 0) and plateau mean (steps 200-1100)
    ax.scatter([0], [-0.141], c='#E53935', s=150, marker='s', zorder=6,
               edgecolors='black', linewidth=1.2, label='Base model')

    # Plateau mean line (steps 200-1100)
    plateau_values = [c[1] for c in llama_checkpoints if c[0] >= 200]
    plateau_mean = np.mean(plateau_values)
    ax.axhline(y=plateau_mean, xmin=200/1180, xmax=1100/1180, color='#4CAF50',
               linewidth=2, linestyle='--', zorder=4, label=f'Plateau mean ({plateau_mean:+.3f})')

    # Zero line
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)

    # Neutral band
    ax.fill_between([min(steps) - 20, max(steps) + 20], -0.05, +0.05,
                    color='#E8F5E9', alpha=0.3, label='Neutral zone')

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
    ax.set_ylabel('Bioalignment Metric ($\\Delta p_{up}$)', fontsize=13)
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


# === FIGURE 4: Before/After in Context of All Models ===

def figure4_context():
    """Figure 4: Bar chart showing all models with Llama 3B and Qwen 3B
    before/after highlighted to show training effect in context."""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create data including bioaligned versions
    # We'll show base models + bioaligned versions side by side for Llama/Qwen
    plot_data = []

    for name, val, sigma, mtype in all_models:
        if name == 'Llama 3B':
            # Add both base and bioaligned
            plot_data.append(('Llama 3B\n(base)', val, sigma, 'intervention-base'))
            plot_data.append(('Llama 3B\n(bioaligned)', llama_bioaligned[1], llama_bioaligned[2], 'intervention-after'))
        elif name == 'Qwen 3B':
            # Add both base and bioaligned
            plot_data.append(('Qwen 3B\n(base)', val, sigma, 'intervention-base'))
            plot_data.append(('Qwen 3B\n(bioaligned)', qwen_bioaligned[1], qwen_bioaligned[2], 'intervention-after'))
        else:
            plot_data.append((name, val, sigma, mtype))

    # Sort by value
    plot_data_sorted = sorted(plot_data, key=lambda x: x[1], reverse=True)

    names = [d[0] for d in plot_data_sorted]
    values = [d[1] for d in plot_data_sorted]
    sigmas = [d[2] for d in plot_data_sorted]
    types = [d[3] for d in plot_data_sorted]

    x_pos = np.arange(len(names))

    # Color coding
    colors = []
    edge_colors = []
    hatches = []
    for v, t in zip(values, types):
        if t == 'intervention-base':
            colors.append('#FFCDD2')  # Light red for base
            edge_colors.append('#D32F2F')
            hatches.append('//')
        elif t == 'intervention-after':
            colors.append('#C8E6C9')  # Light green for after
            edge_colors.append('#388E3C')
            hatches.append('')
        elif v > 0.05:
            colors.append('#1976D2')
            edge_colors.append('black')
            hatches.append('')
        elif v < -0.05:
            colors.append('#D32F2F')
            edge_colors.append('black')
            hatches.append('')
        else:
            colors.append('#757575')
            edge_colors.append('black')
            hatches.append('')

    # Create bars (no error bars - sigma values are in tables)
    bars = ax.bar(x_pos, values, width=0.7, color=colors, edgecolor=edge_colors, linewidth=1.2)

    # Add hatching
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Zero line
    ax.axhline(y=0, color='black', linewidth=1.0, linestyle='-')

    # Neutral zone
    ax.axhspan(-0.05, 0.05, color='#E8E8E8', alpha=0.3, zorder=0)

    # Add curved arrows connecting base to bioaligned for Llama and Qwen
    from matplotlib.patches import FancyArrowPatch

    llama_base_idx = names.index('Llama 3B\n(base)')
    llama_bio_idx = names.index('Llama 3B\n(bioaligned)')
    qwen_base_idx = names.index('Qwen 3B\n(base)')
    qwen_bio_idx = names.index('Qwen 3B\n(bioaligned)')

    # Llama curved arrow (arc upward toward top of plot)
    llama_start_y = max(values[llama_base_idx], values[llama_bio_idx]) + 0.02
    llama_arrow = FancyArrowPatch(
        (llama_base_idx, llama_start_y),
        (llama_bio_idx, llama_start_y),
        connectionstyle='arc3,rad=0.5',  # Positive rad = arc upward
        arrowstyle='->,head_length=8,head_width=5',
        color='#1976D2', linewidth=2.5, zorder=10
    )
    ax.add_patch(llama_arrow)
    # Position label above the arc peak
    llama_shift = llama_bioaligned[1] - llama_base[1]
    ax.text((llama_base_idx + llama_bio_idx)/2, llama_start_y + 0.06, f'+{llama_shift:.3f}', ha='center',
            fontsize=12, fontweight='bold', color='#1976D2',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1976D2', alpha=0.95))

    # Qwen curved arrow (arc upward toward top of plot)
    qwen_start_y = max(values[qwen_base_idx], values[qwen_bio_idx]) + 0.02
    qwen_arrow = FancyArrowPatch(
        (qwen_base_idx, qwen_start_y),
        (qwen_bio_idx, qwen_start_y),
        connectionstyle='arc3,rad=0.5',  # Positive rad = arc upward
        arrowstyle='->,head_length=8,head_width=5',
        color='#E65100', linewidth=2.5, zorder=10
    )
    ax.add_patch(qwen_arrow)
    # Position label above the arc peak
    qwen_shift = qwen_bioaligned[1] - qwen_base[1]
    ax.text((qwen_base_idx + qwen_bio_idx)/2, qwen_start_y + 0.06, f'+{qwen_shift:.3f}', ha='center',
            fontsize=12, fontweight='bold', color='#E65100',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E65100', alpha=0.95))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10, rotation=45, ha='right')
    ax.set_ylabel('Bioalignment Metric ($\\Delta p_{up}$)', fontsize=13)
    ax.set_ylim(-0.18, 0.30)  # Adjusted for no error bars
    ax.set_title('Effect of Bioaligned Training in Context of All Models',
                 fontsize=14, fontweight='bold', pad=12)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FFCDD2', edgecolor='#D32F2F', hatch='//', label='Base (before training)'),
        Patch(facecolor='#C8E6C9', edgecolor='#388E3C', label='Bioaligned (after training)'),
        Patch(facecolor='#1976D2', edgecolor='black', label='Pro-biological'),
        Patch(facecolor='#757575', edgecolor='black', label='Neutral'),
        Patch(facecolor='#D32F2F', edgecolor='black', label='Pro-synthetic'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              framealpha=0.95, edgecolor='gray')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, 'figure4_context.png')
    fig.savefig(path)
    path_pdf = os.path.join(FIGURES_DIR, 'figure4_context.pdf')
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f'Figure 4 saved: {path}')


# === Generate all figures ===

if __name__ == '__main__':
    print('Generating figures...')
    figure1_bar()       # Horizontal bar chart of all models
    figure2_bar()       # Before/after bar chart (headline result)
    figure3_dynamics()  # Training dynamics
    figure4_context()   # Full context with intervention
    print(f'\nAll figures saved to: {FIGURES_DIR}')
