from pathlib import Path
import subprocess, sys
HERE=Path(__file__).resolve().parent
for script in ['01_extract_azimuth_reference.py','02a_prepare_azimuth_descriptors.py','02_fresh68k_ablation_and_entropy_case.py','03_azimuth_strict_5fold_pca_apg.py','04_azimuth_group_sensitivity.py','05_make_figures.py']:
    print(f'Running {script}', flush=True)
    subprocess.run([sys.executable,str(HERE/script)],check=True)
