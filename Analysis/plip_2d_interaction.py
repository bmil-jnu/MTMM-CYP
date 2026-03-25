"""
PLIP 2D Interaction Diagram Generator
LCZ-CYP3A4 and LCZ-CYP1A2
Based on PLIP v2.4.0 analysis results (AutoDock4 top-ranked poses)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams['font.family'] = 'Times New Roman'


def draw_interaction_diagram(ax, title, interactions, colors):
    """Draw a 2D ligand-protein interaction diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Central ligand ellipse
    lig_ellipse = plt.Circle((5, 5), 0.8, color='gold', zorder=5)
    ax.add_patch(lig_ellipse)
    ax.text(5, 5, 'LCZ', ha='center', va='center',
            fontsize=12, fontweight='bold', zorder=6)

    line_styles = {
        'Hydrophobic': '--',
        'H-bond':      '-',
        'pi-Stack':    '-.',
        'Halogen':     ':',
        'pi-Cation':   '-.'
    }

    for itype, label, x, y, dist in interactions:
        color = colors[itype]
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color,
                    alpha=0.85, edgecolor='gray')
        ax.text(x, y, f'{label}\n({dist})', ha='center', va='center',
                fontsize=8.5, bbox=bbox, zorder=5)

        dx = 5 - x
        dy = 5 - y
        length = np.sqrt(dx**2 + dy**2)
        ax.annotate('',
                    xy=(5 - dx * 0.8 / length, 5 - dy * 0.8 / length),
                    xytext=(x + dx * 0.45, y + dy * 0.45),
                    arrowprops=dict(
                        arrowstyle='-',
                        color=colors[itype],
                        lw=1.8,
                        linestyle=line_styles[itype]
                    ))

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)


def plot_cyp3a4_interaction(output_path):
    """
    LCZ-CYP3A4 PLIP interaction diagram
    PLIP v2.4.0 results:
    - Hydrophobic (9): Phe108, Ile301, Phe304, Ala305, Thr309,
                       Ala370, Ile120, Phe241, Thr118
    - H-bond (1): Ser119 (3.68 Å)
    - pi-Cation (1): Arg212 (3.96 Å)
    """
    colors = {
        'Hydrophobic': '#FFAAAA',
        'H-bond':      '#AACCFF',
        'pi-Cation':   '#AAFFAA',
    }

    interactions = [
        # (type, label, x, y, distance)
        ('Hydrophobic', 'Phe304',  3.5, 7.5, '3.01 Å'),
        ('Hydrophobic', 'Phe108',  2.0, 6.0, '3.14 Å'),
        ('Hydrophobic', 'Ile301',  2.5, 4.0, '2.97 Å'),
        ('Hydrophobic', 'Ala305',  3.5, 2.5, '3.20 Å'),
        ('Hydrophobic', 'Thr309',  5.5, 2.0, '3.57 Å'),
        ('Hydrophobic', 'Ala370',  7.5, 2.5, '3.10 Å'),
        ('Hydrophobic', 'Ile120',  8.0, 4.5, '3.24 Å'),
        ('Hydrophobic', 'Phe241',  8.0, 6.5, '3.85 Å'),
        ('Hydrophobic', 'Thr118',  7.0, 8.0, '3.39 Å'),
        ('H-bond',      'Ser119',  5.5, 8.5, '3.68 Å'),
        ('pi-Cation',   'Arg212',  3.5, 8.5, '3.96 Å'),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    draw_interaction_diagram(
        ax,
        'LCZ–CYP3A4 Protein-Ligand Interactions\n(AutoDock4, PLIP v2.4.0)',
        interactions,
        colors
    )

    patches = [
        mpatches.Patch(color='#FFAAAA', label='Hydrophobic (9)'),
        mpatches.Patch(color='#AACCFF', label='H-bond (1)'),
        mpatches.Patch(color='#AAFFAA', label='π-Cation (1)'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=10,
              framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✅ CYP3A4 PLIP 2D diagram saved: {output_path}')


def plot_cyp1a2_interaction(output_path):
    """
    LCZ-CYP1A2 PLIP interaction diagram
    PLIP v2.4.0 results:
    - Hydrophobic (9): Phe260, Phe319, Asp320, Thr321, Thr223,
                       Leu497, Ile386, Thr118, Thr124
    - H-bond (2): Val227 (4.03 Å), Asp320 (4.06 Å)
    - pi-Stacking (1): Phe226 (3.90 Å)
    - Halogen bond (1): Asp313-Cl (3.61 Å)
    """
    colors = {
        'Hydrophobic': '#FFAAAA',
        'H-bond':      '#AACCFF',
        'pi-Stack':    '#AAFFAA',
        'Halogen':     '#FFDDAA',
    }

    interactions = [
        # (type, label, x, y, distance)
        ('Hydrophobic', 'Phe260',     2.0, 7.5, '2.77 Å'),
        ('Hydrophobic', 'Phe319',     2.0, 5.5, '2.78 Å'),
        ('Hydrophobic', 'Asp320',     2.0, 4.0, '2.79 Å'),
        ('Hydrophobic', 'Thr321',     2.5, 2.5, '2.81 Å'),
        ('Hydrophobic', 'Thr223',     4.0, 1.5, '3.07 Å'),
        ('Hydrophobic', 'Leu497',     6.0, 1.5, '3.08 Å'),
        ('Hydrophobic', 'Ile386',     7.5, 2.5, '3.13 Å'),
        ('Hydrophobic', 'Thr118',     8.5, 4.5, '3.39 Å'),
        ('Hydrophobic', 'Thr124',     8.0, 6.5, '3.51 Å'),
        ('H-bond',      'Val227',     6.5, 8.5, '4.03 Å'),
        ('H-bond',      'Asp320*',    4.5, 8.5, '4.06 Å'),
        ('pi-Stack',    'Phe226',     3.0, 8.5, '3.90 Å'),
        ('Halogen',     'Asp313\n(Cl)', 8.0, 8.0, '3.61 Å'),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    draw_interaction_diagram(
        ax,
        'LCZ–CYP1A2 Protein-Ligand Interactions\n(AutoDock4, PLIP v2.4.0)',
        interactions,
        colors
    )

    patches = [
        mpatches.Patch(color='#FFAAAA', label='Hydrophobic (9)'),
        mpatches.Patch(color='#AACCFF', label='H-bond (2)'),
        mpatches.Patch(color='#AAFFAA', label='π-Stacking (1)'),
        mpatches.Patch(color='#FFDDAA', label='Halogen bond (1)'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=10,
              framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✅ CYP1A2 PLIP 2D diagram saved: {output_path}')


if __name__ == '__main__':
    output_dir = '/mnt/c/Users/user/Downloads/docking_viz'
    os.makedirs(output_dir, exist_ok=True)

    plot_cyp3a4_interaction(
        os.path.join(output_dir, 'LCZ_CYP3A4_PLIP_2D.png')
    )
    plot_cyp1a2_interaction(
        os.path.join(output_dir, 'LCZ_CYP1A2_PLIP_2D.png')
    )
    print('✅ All PLIP 2D interaction diagrams saved.')
