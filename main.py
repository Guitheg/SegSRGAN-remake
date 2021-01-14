from os.path import abspath, normpath, dirname, join

MAIN_PATH = normpath(abspath(dirname(__file__)))
INIT_CONFIG_INI_FILEPATH = abspath(normpath(join(MAIN_PATH, "init_config.ini")))
CONFIG_INI_FILEPATH = abspath(normpath(join(MAIN_PATH, "config.ini")))