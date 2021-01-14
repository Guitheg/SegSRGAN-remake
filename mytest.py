import importlib
import sys
from utils.config_init import get_configparser
from main import CONFIG_INI_FILEPATH
import importlib
import argparse

def main():
    
    config = get_configparser(CONFIG_INI_FILEPATH)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help = "name of test python file (sans le .py)", required=True)
    parser.add_argument("-p", "--params", help="list of parameters", nargs="+")
    args = parser.parse_args()
    
    module_path = f"tests.{str(args.filename).split('.')[0]}"
    print(f"Test : {module_path}")
    runtest_fct = importlib.import_module(module_path).runtest
    runtest_fct(config, args.params)
    
    sys.exit(0)

if __name__ == "__main__":
    main()