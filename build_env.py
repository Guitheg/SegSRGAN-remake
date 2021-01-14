from configparser import ConfigParser
from utils.files import get_environment
from main import INIT_CONFIG_INI_FILEPATH, CONFIG_INI_FILEPATH
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--home_folder", "-f", help = "Home folder where datas will be generated etc.", required=True)
    
    args = parser.parse_args()
    
    config = ConfigParser()
    config.read(INIT_CONFIG_INI_FILEPATH)
    
    (home_path, _, _,  _, _, _, _, _, _, _) = get_environment(args.home_folder, config)
    
    config.set('Path', 'Home', value=home_path)
    
    with open(CONFIG_INI_FILEPATH, 'w') as configfile:
        config.write(configfile)
        
    sys.exit(0)
    
if __name__ == "__main__":
    main()