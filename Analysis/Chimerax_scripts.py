"""
ChimeraX 3D Docking Pose Script Generator
LCZ-CYP3A4 and LCZ-CYP1A2 (AutoDock4 best poses)

This script generates .cxc files to be opened in UCSF ChimeraX.

Usage:
  python chimerax_scripts.py

Then in ChimeraX Command Line:
  open C:/Users/user/Downloads/docking_viz/viz_CYP3A4_docking.cxc
  open C:/Users/user/Downloads/docking_viz/viz_CYP1A2_docking.cxc
"""

import os

OUTPUT_DIR = '/mnt/c/Users/user/Downloads/docking_viz'


# ── CYP3A4 Configuration ──────────────────────────────────────────────────────
CYP3A4_CONFIG = {
    'pdb_file':   'CYP3A4_AD4_complex.pdb',
    'output_cxc': 'viz_CYP3A4_docking.cxc',
    'output_png': 'CYP3A4_AD4_final.png',

    # HEM: chain B, residue 508
    # LCZ: chain d, residue 1 (UNL)
    'hem_sel':  '#1/B:508',
    'lcz_sel':  '#1/d:1',

    # Fe label atom
    'fe_atom':  '#1/B:508@FE',

    # Key active-site residues to display
    # CYP3A4: Arg105, Phe304, Thr309, Ala305, Ala370
    'active_site_res': '#1/A:105,213,301,302,304,305,309,370,442,445',

    # Residue labels (residue_number: label_text)
    'labels': {
        '#1/A:304': 'Phe304',
        '#1/A:309': 'Thr309',
        '#1/A:105': 'Arg105',
    },

    # Fe-LCZ distance measurement
    # Closest atom: Serial 10 (C) = 3.054 Å
    'distance_atom': '#1/d:1@C10',
    'fe_dist_label': 'Fe–C: 3.05 Å',

    'fe_coords': (-15.846, -23.032, -11.293),
}

# ── CYP1A2 Configuration ──────────────────────────────────────────────────────
CYP1A2_CONFIG = {
    'pdb_file':   'CYP1A2_AD4_complex.pdb',
    'output_cxc': 'viz_CYP1A2_docking.cxc',
    'output_png': 'CYP1A2_AD4_final.png',

    # HEM: chain B, residue 900
    # LCZ: chain d, residue 1 (UNL)
    'hem_sel':  '#1/B:900',
    'lcz_sel':  '#1/d:1',

    # Fe label atom
    'fe_atom':  '#1/B:900@FE',

    # Key active-site residues
    # CYP1A2: Phe125, Phe226, Thr321, Ile386, Phe381, Phe384
    'active_site_res': '#1/A:125,224,257,312,320,321,382',

    # Residue labels
    'labels': {
        '#1/A:226': 'Phe226',
        '#1/A:321': 'Thr321',
        '#1/A:125': 'Phe125',
    },

    # Fe-LCZ distance measurement
    # Closest atom: Serial 22 (C) = 3.425 Å
    'distance_atom': '#1/d:1@C22',
    'fe_dist_label': 'Fe–C: 3.43 Å',

    'fe_coords': (4.916, 24.997, 23.908),
}


# ── Script Generator ──────────────────────────────────────────────────────────

def generate_chimerax_script(cfg, output_dir):
    """Generate a ChimeraX .cxc command script for a docking complex."""

    lines = []

    # --- Open and basic display ---
    lines += [
        f'# ChimeraX visualization script',
        f'# Generated for: {cfg["pdb_file"]}',
        f'',
        f'open {cfg["pdb_file"]}',
        f'set bgColor white',
        f'lighting simple',
        f'graphics silhouettes false',
        f'',
        f'# Protein cartoon (gray)',
        f'cartoon #1',
        f'color #1 #AAAAAA',
        f'hide #1 atoms',
        f'',
    ]

    # --- HEM (cyan) ---
    lines += [
        f'# Heme cofactor (cyan)',
        f'select {cfg["hem_sel"]}',
        f'show sel atoms',
        f'style sel stick',
        f'color sel cyan',
        f'select clear',
        f'',
    ]

    # --- LCZ (yellow, element-colored heteroatoms) ---
    lines += [
        f'# LCZ ligand (yellow + element colors)',
        f'select {cfg["lcz_sel"]}',
        f'show sel atoms',
        f'style sel stick',
        f'color sel yellow',
        f'color sel & N blue',
        f'color sel & O red',
        f'color sel & Cl green',
        f'select clear',
        f'',
    ]

    # --- Active site residues (lightpink) ---
    lines += [
        f'# Active-site residues (pink)',
        f'show {cfg["active_site_res"]} atoms',
        f'style {cfg["active_site_res"]} stick',
        f'color {cfg["active_site_res"]} lightpink',
        f'',
    ]

    # --- Camera view ---
    lines += [
        f'# Center view on heme',
        f'view {cfg["hem_sel"]}',
        f'',
    ]

    # --- Labels ---
    lines += [f'# Residue and Fe labels']
    lines += [
        f'label {cfg["fe_atom"]} atoms text "Fe" color darkorange size 16'
    ]
    for sel, text in cfg['labels'].items():
        lines += [
            f'label {sel} residues text "{text}" color darkblue size 12'
        ]
    lines += ['']

    # --- Fe-LCZ distance ---
    lines += [
        f'# Fe–ligand minimum distance ({cfg["fe_dist_label"]})',
        f'mousemode right distance',
        f'# Right-click Fe atom, then right-click nearest LCZ carbon',
        f'# Or run: distance {cfg["fe_atom"]} {cfg["distance_atom"]}',
        f'',
    ]

    # --- Save image ---
    lines += [
        f'# Save high-resolution image',
        f'# Adjust the view angle first, then run:',
        f'# save {cfg["output_png"]} width 1600 height 1200 supersample 4',
        f'',
    ]

    # Write file
    out_path = os.path.join(output_dir, cfg['output_cxc'])
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'✅ ChimeraX script saved: {out_path}')


def print_usage():
    print('\n' + '='*60)
    print('ChimeraX scripts generated!')
    print('='*60)
    print('\nIn ChimeraX Command Line, run:')
    print(f'  open C:/Users/user/Downloads/docking_viz/viz_CYP3A4_docking.cxc')
    print(f'  open C:/Users/user/Downloads/docking_viz/viz_CYP1A2_docking.cxc')
    print('\nAfter adjusting the view angle, save with:')
    print('  save CYP3A4_AD4_final.png width 1600 height 1200 supersample 4')
    print('  save CYP1A2_AD4_final.png width 1600 height 1200 supersample 4')
    print('\nFe–ligand distances:')
    print('  CYP3A4: Fe–C = 3.054 Å (Serial 10, aromatic carbon)')
    print('  CYP1A2: Fe–C = 3.425 Å (Serial 22, aromatic carbon)')
    print('\n⚠️  Note: Fe–C distances reflect minimum Fe–ligand distances')
    print('   (not Fe–N coordination) due to AutoDock4 force field limitations.')


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generate_chimerax_script(CYP3A4_CONFIG, OUTPUT_DIR)
    generate_chimerax_script(CYP1A2_CONFIG, OUTPUT_DIR)
    print_usage()
