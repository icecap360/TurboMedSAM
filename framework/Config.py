import importlib
import os
from importlib.machinery import SourceFileLoader

def import_module(module_name, module_path):
    # return importlib.import_module( path, package=pkg )
    return SourceFileLoader(module_name, module_path).load_module()
    spec = importlib.util.spec_from_file_location(module_name, module_path) 
    return importlib.util.module_from_spec(spec)
def read_cfg_str( path):
    with open(path) as reader:
        string = reader.readlines()
    return string