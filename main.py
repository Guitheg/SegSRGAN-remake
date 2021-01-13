from os.path import abspath, normpath, dirname, join

MAIN_PATH = normpath(abspath(dirname(__file__)))
CONFIG_INI_FILEPATH = abspath(normpath(join(MAIN_PATH, "config.ini")))
DATA_EXAMPLE = abspath(normpath(join(MAIN_PATH, "data_example")))