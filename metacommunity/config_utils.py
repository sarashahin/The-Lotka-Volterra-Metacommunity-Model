############################################
# Configuration File Handling
############################################

import importlib
import inspect

def load_config(config_file_path):
    """
    Loads a configuration from a .py file.
    """
    try:
        module_name = config_file_path[:-3] if config_file_path.endswith('.py') else config_file_path
        module_name = module_name.replace("/", ".")
        config_module = importlib.import_module(module_name)
        return config_module
    except (ImportError, FileNotFoundError) as e:
        print(f"Error loading config file: {e}")
        return False

def assign_all_config_variables(config_module, target_globals, verbose=False):
    """
    Assigns all variables from config_module to the target_globals.
    """
    for name, value in config_module.__dict__.items():
        if not name.startswith("__"):
            if verbose:
                print(f"{name} = {value}")
            target_globals[name] = value

def configure_modules(config_module, modules_to_configure):
    """
    Assigns config_module variables to all modules in modules_to_configure.
    """
    for module in modules_to_configure:
        if inspect.ismodule(module):
            assign_all_config_variables(config_module, module.__dict__)
