"""
    srun --mem-per-cpu=16G --time=00:20:00 --gpus-per-node=1\
            python finalized_scripts/compare_integrators_nlse_2d.py\
            --exe1 build/bin/nlse_2d_dev --name1 "SS2"\
            --exe2 build/bin/nlse_sewi_2d_dev --name2 "sEWI"\
            --output_dir $SCRATCH/integrator_study_nlse\
            --T_sim_study .8 --nx_values_study 200 400 800\
            --dt_values_study  .01 0.005 .001\
            --max_snapshots_study 49  --anisotropy_type_study=$c
"""

import argparse
import sys
from pathlib import Path
import json
from compare_utils_complex_2d import NLSEIntegratorStudy

def main():
    parser = argparse.ArgumentParser(description="Run Generalized NLSE integrator comparison studies.")
    parser.add_argument('--exe1',
            type=str, help="Path to the first integrator executable.")
    parser.add_argument('--name1',
            type=str, help="Name of the first integrator.")
    parser.add_argument('--exe2',
            type=str, help="Path to the second integrator executable.")
    parser.add_argument('--name2',
            type=str, help="Name of the second integrator.")
    parser.add_argument('--output_dir',
            type=str, required=True, help="Base directory for study outputs.")
    parser.add_argument('--L_study',
            type=float, default=10., help="Domain half-length L for study.")
    parser.add_argument('--T_sim_study',
            type=float, default=1.2, help="Total simulation time T for study.")
    parser.add_argument('--nx_values_study',
            type=int, nargs='+', default=[128, 256], help="Nx values for spatial resolution. Highest is used for base IC generation.")
    parser.add_argument('--dt_values_study',
            type=float, nargs='+', default=[0.01, 0.005, 0.0025], help="dt values for temporal resolution for the study.")
    parser.add_argument('--ic_type_study',
            type=str, default="colliding_packets_nlse", # FYI this one's hardcoded.
            help="Phenomenon type for IC. Generated once at highest Nx and downsampled.")
    parser.add_argument('--phenomenon_params_override_json_study',
            type=str, default="{}", help="JSON string of parameters for ic_type_study. Example: '{\"amplitude1\": 1.0}'")
    parser.add_argument('--m_type_study',
            type=str, default="uniform_unity_attractive", help="Type of m(x,y) field for NLSE. Generated once at highest Nx and downsampled.")
    parser.add_argument('--anisotropy_type_study',
            type=str, default="constant", help="Type of c(x,y) field for NLSE. Generated once at highest Nx and downsampled.")
    parser.add_argument('--downsampling_strategy_study',
            type=str, choices=['FFT', 'interpolation'], default='interpolation', help="Strategy for downsampling ICs/fields.")
    parser.add_argument('--max_snapshots_study',
            type=int, default=50, help="Maximum number of snapshots to save per run in the study.")
    parser.add_argument('--keep_temps', action='store_true', help="Keep temporary files after runs.")
    parser.add_argument('--force_regenerate_ics', action='store_true', help="Force regeneration of downsampled initial conditions.")

    args = parser.parse_args()

    if not ((args.exe1 and args.name1) or (args.exe2 and args.name2)):
        parser.error("At least one integrator (exe and name) must be specified for the study.")
    if args.exe1 and not Path(args.exe1).exists():
        parser.error(f"Executable for integrator 1 not found: {args.exe1}")
    if args.exe2 and not Path(args.exe2).exists():
        parser.error(f"Executable for integrator 2 not found: {args.exe2}")
    try:
        json.loads(args.phenomenon_params_override_json_study)
    except json.JSONDecodeError as e:
        parser.error(f"Invalid JSON string for phenomenon_params_override_json_study: {e}")

    study_params_config = {
        'L_study': args.L_study,
        'T_sim_study': args.T_sim_study,
        'nx_values_study': sorted(list(set(args.nx_values_study))),
        'dt_values_study': sorted(list(set(args.dt_values_study))),
        'ic_type_study': args.ic_type_study,
        'phenomenon_params_override_json_study': args.phenomenon_params_override_json_study,
        'm_type_study': args.m_type_study,
        'anisotropy_type_study': args.anisotropy_type_study,
        'downsampling_strategy_study': args.downsampling_strategy_study,
        'max_snapshots_study': args.max_snapshots_study
    }
    study_runner = NLSEIntegratorStudy(args, study_params_config)
    study_runner.execute()

if __name__ == '__main__':
    sys.exit(main())
