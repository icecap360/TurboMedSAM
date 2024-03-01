import importlib


def import_module(path):
    return importlib.import_module(path)
def read_cfg_str( path):
    with open(path) as reader:
        string = reader.readlines()
    return string