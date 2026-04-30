import importlib
import inspect  # For checking if it's a module

############################################################
# Handling of configuration files 
############################################################

def load_config(config_file_path):
    """
    Loads a configuration from a .py file and assigns variables to the global scope.
    """
    try:
        module_name = config_file_path[:-3] if config_file_path.endswith('.py') else config_file_path
        module_name = module_name.replace("/", ".")  # Handle paths
        config_module = importlib.import_module(module_name)
        return config_module  # Indicate successful loading

    except (ImportError, FileNotFoundError) as e:
        print(f"Error loading config file: {e}")
        return False

def assign_all_config_variables(config_module, target_globals, verbose=False):
    """
    Assign variables as given in config_module in the global context
    given as target_globals.
    """
    for name, value in config_module.__dict__.items():
        if not name.startswith("__"):
            if verbose:
                print(f"{name} = {value}")
            target_globals[name] = value  # Assign to the target globals

def configure_modules(config_module, modules_to_configure):
    """
    Assign variables as per config_module in all modules_to_configure.
    """
    for module in modules_to_configure:
        if inspect.ismodule(module): # Check if its really a module
            assign_all_config_variables(config_module, module.__dict__)


