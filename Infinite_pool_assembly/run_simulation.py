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
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
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

############################################################
# Logging Setup
############################################################
def setup_logging_custom():
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = RotatingFileHandler(
        os.path.join(log_folder, "debug.log"),
        maxBytes=100000,
        backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


if __name__ == '__main__':
    setup_logging_custom()
    logger = logging.getLogger(__name__)
    logger.info("Logging is set up. Debug logs will be stored in the 'logs' folder.")
    logger.debug(f"################ NEW RUN ################")
    main()
