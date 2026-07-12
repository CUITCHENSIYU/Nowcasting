import glob
from importlib.util import spec_from_file_location, module_from_spec
import os
import warnings
import traceback

__all__ = ["register_module", "get_module"]


# mapping of string names to module names
_all_modules = {
    "trainers": {},
    "input_pipelines": {},
    "evaluators": {},
    "backbone": {},
    "head": {},
    "models": {},
    "criterions": {},
    "runners": {},
    "datasets": {},
    "pipelines": {},
}


def register_module(*args, **kwargs):
    def _register(func):
        parent_name = kwargs["parent"]
        func_name = func.__name__
        _all_modules[parent_name][func_name] = func
        return func
    return _register


def get_module(parent, name):
    assert name in _all_modules[parent], \
            "{} is not found in {} registry, all supported names: {}".format(name, parent, list(_all_modules[parent].keys()))
    return _all_modules[parent][name]

def load_modules(root):
    module_paths = find_module_paths(root)
    for module_path in module_paths:
        module_name = os.path.splitext(os.path.relpath(module_path, root))[0].replace(os.sep, ".")
        try:
            spec = spec_from_file_location(module_name, module_path)
            m = module_from_spec(spec=spec)
            spec.loader.exec_module(m)
        except Exception:
            warnings.warn(
                f"Failed to load module: {module_path}, "
                f"trace: {traceback.format_exc()}"
            )

def find_module_paths(root):
    module_paths = [
        f
        for f in glob.glob(os.path.join(root, "**/*.py"), recursive=True)
        if os.path.isfile(f) and (not f.endswith("__init__.py"))
    ]
    return module_paths