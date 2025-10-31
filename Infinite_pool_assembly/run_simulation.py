############################################
# run_simulation.py
############################################
"""
Main entry point for running ecological simulations.

This script parses the desired simulation engine ('ibm', 'psd2', 'ode')
from the command line and delegates the entire workflow to the
corresponding SimulationRunner subclass.

Examples:
  # Run infinite pool assembly with the IBM engine
  python run_simulation.py --engine ibm --pool 120

  # Run a 3-species RPS benchmark with the PSD2 engine
  python run_simulation.py --engine psd2
"""
import sys
import argparse
from simulation_runner import IBMSimulation, PSD2Simulation, ODESimulation

def main():
    """Parses CLI args and launches the appropriate simulation runner."""
    # Use argparse to find the engine, which is more robust than manual parsing.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--engine', type=str, choices=['ibm', 'psd2', 'ode'], default='ibm')
    args, _ = parser.parse_known_args()

    engine_map = {
        'ibm': IBMSimulation,
        'psd2': PSD2Simulation,
        'ode': ODESimulation
    }

    if args.engine in engine_map:
        # Pass all command-line arguments to the selected runner's constructor
        runner = engine_map[args.engine]()
        runner.execute()
    else:
        # This case should not be reached due to argparse 'choices'
        print(f"Error: Unknown engine '{args.engine}'.", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
