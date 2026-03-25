"""
MD Simulation Figure Generator
LCZ-CYP3A4 (50 ns) and LCZ-CYP1A2 (50 ns)
Generates: Backbone RMSD + per-residue Cα RMSF panels

Input files required:
  CYP3A4: ~/docking_project/MD/LCZ_CYP3A4_v3/rmsd_protein.xvg
           ~/docking_project/MD/LCZ_CYP3A4_v3/rmsf.xvg
  CYP1A2: ~/docking_project/MD/LCZ_CYP1A2/rmsd_protein.xvg
           ~/docking_project/MD/LCZ_CYP1A2/rmsf.xvg
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.expanduser('~/docking_project/MD')
OUTPUT_DIR  = '/mnt/c/Users/user/Downloads/docking_viz'

PATHS = {
    'CYP3A4': {
        'rmsd': os.path.join(BASE_DIR, 'LCZ_CYP3A4_v3', 'rmsd_protein.xvg'),
        'rmsf': os.path.join(BASE_DIR, 'LCZ_CYP3A4_v3', 'rmsf.xvg'),
        'color': 'steelblue',
        'equil_start': 20,   # ns — post-equilibration starts here
        'rmsf_regions': [
            # (start_res, end_res, color, label, label_pos_x)
            (185, 210, 'orange', 'F-G loop',  197),
            (255, 275, 'purple', 'I-helix',   265),
        ],
        'output': 'MD_CYP3A4_50ns_final.png',
        'suptitle': 'MD Simulation Analysis: LCZ–CYP3A4 Complex (50 ns, 310 K)',
        'rmsd_title': 'CYP3A4 Protein Backbone RMSD (50 ns)',
        'rmsf_title': 'CYP3A4 Residue Flexibility (RMSF, Cα)',
    },
    'CYP1A2': {
        'rmsd': os.path.join(BASE_DIR, 'LCZ_CYP1A2', 'rmsd_protein.xvg'),
        'rmsf': os.path.join(BASE_DIR, 'LCZ_CYP1A2', 'rmsf.xvg'),
        'color': 'darkorange',
        'equil_start': 20,
        'rmsf_regions': [
            (215, 235, 'green', 'F-helix', 225),
        ],
        'rmsf_sublabels': {
            225: '(Substrate\nrecognition)'
        },
        'output': 'MD_CYP1A2_50ns_final.png',
        'suptitle': 'MD Simulation Analysis: LCZ–CYP1A2 Complex (50 ns, 310 K)',
        'rmsd_title': 'CYP1A2 Protein Backbone RMSD (50 ns)',
        'rmsf_title': 'CYP1A2 Residue Flexibility (RMSF, Cα)',
    },
}

# ── Plot functions ────────────────────────────────────────────────────────────

def plot_rmsd(ax, time, rmsd, color, title, equil_start):
    """Plot backbone RMSD with post-equilibration region."""
    mean_val = np.mean(rmsd)
    std_val  = np.std(rmsd)

    ax.plot(time, rmsd, color=color, linewidth=1.0, alpha=0.9, label='Backbone RMSD')
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_val:.2f} ± {std_val:.2f} Å')
    ax.fill_between(time, rmsd, alpha=0.15, color=color)
    ax.axvspan(equil_start, max(time), alpha=0.08, color='green')
    ax.text(
        equil_start + (max(time) - equil_start) * 0.5,
        max(rmsd) * 0.92,
        'Post-equilibration region',
        fontsize=12, color='darkgreen', fontweight='bold', ha='center'
    )

    ax.set_xlabel('Time (ns)', fontsize=15)
    ax.set_ylabel('RMSD (Å)', fontsize=15)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0, max(time)])
    ax.tick_params(labelsize=13)

    print(f'  RMSD: {mean_val:.2f} ± {std_val:.2f} Å  '
          f'(min={np.min(rmsd):.2f}, max={np.max(rmsd):.2f})')


def plot_rmsf(ax, res, rmsf, color, title, regions, sublabels=None):
    """Plot per-residue Cα RMSF with annotated functional regions."""
    sublabels = sublabels or {}
    ymax = max(rmsf)

    ax.plot(res, rmsf, color=color, linewidth=1.0, alpha=0.8)
    ax.fill_between(res, rmsf, alpha=0.15, color=color)
    ax.fill_between(res, rmsf, where=rmsf > 2.0, alpha=0.4,
                    color='salmon', label='High flexibility (>2 Å)')
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=1.0,
               alpha=0.6, label='Threshold: 2 Å')

    for (r_start, r_end, reg_color, label, lx) in regions:
        ax.axvspan(r_start, r_end, alpha=0.15, color=reg_color)
        ax.text(lx, ymax * 0.97, label,
                fontsize=11, color=reg_color, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.7, edgecolor=reg_color))
        if lx in sublabels:
            ax.text(lx, ymax * 0.82, sublabels[lx],
                    fontsize=9, color=reg_color, ha='center')

    ax.set_xlabel('Residue Number', fontsize=15)
    ax.set_ylabel('RMSF (Å)', fontsize=15)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=13)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([min(res), max(res)])
    ax.tick_params(labelsize=13)

    top_idx = np.argmax(rmsf)
    print(f'  RMSF max: Res {int(res[top_idx])} ({rmsf[top_idx]:.2f} Å)')


def generate_md_figure(cyp_name, cfg, output_dir):
    """Generate RMSD + RMSF figure for a given CYP system."""
    print(f'\n[{cyp_name}] Generating MD figure...')

    # Load data
    rmsd_data = np.loadtxt(cfg['rmsd'], comments=['#', '@'])
    rmsf_data = np.loadtxt(cfg['rmsf'], comments=['#', '@'])

    time = rmsd_data[:, 0]
    rmsd = rmsd_data[:, 1] * 10   # nm → Å
    res  = rmsf_data[:, 0]
    rmsf = rmsf_data[:, 1] * 10   # nm → Å

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    gs  = gridspec.GridSpec(2, 1, hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    plot_rmsd(ax1, time, rmsd,
              cfg['color'], cfg['rmsd_title'], cfg['equil_start'])

    plot_rmsf(ax2, res, rmsf,
              cfg['color'], cfg['rmsf_title'],
              cfg['rmsf_regions'],
              cfg.get('rmsf_sublabels', {}))

    fig.suptitle(cfg['suptitle'], fontsize=16, fontweight='bold', y=1.01)

    out_path = os.path.join(output_dir, cfg['output'])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✅ Saved: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cyp_name, cfg in PATHS.items():
        # Check input files
        missing = [f for f in [cfg['rmsd'], cfg['rmsf']] if not os.path.exists(f)]
        if missing:
            print(f'[{cyp_name}] ⚠️  Missing files: {missing}')
            continue
        generate_md_figure(cyp_name, cfg, OUTPUT_DIR)

    print('\n✅ All MD figures saved.')
