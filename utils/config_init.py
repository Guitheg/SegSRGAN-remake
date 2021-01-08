
import configparser
from os.path import isfile, normpath


def get_configparser(config_ini_filepath : str, verbose : bool = False):
    filepath = normpath(config_ini_filepath)
    if verbose :
        print(f"Chargement de {filepath}...")
    
    if not isfile(filepath):
        raise Exception(f"Le chemin vers le fichier '{filepath}' est introuvable")
    
    config = configparser.ConfigParser()
    config.read(filepath)
    return config