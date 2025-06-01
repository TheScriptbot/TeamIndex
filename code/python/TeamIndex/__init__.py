import importlib

def __getattr__(name):
    if name == 'creation':
        return importlib.import_module('.creation', package=__name__)
    elif name == 'evaluation':
        return importlib.import_module('.evaluation', package=__name__)
    elif name == 'benchmark':
        return importlib.import_module('.benchmark', package=__name__)
    elif name == 'generation':
        return importlib.import_module('.generation', package=__name__)
    elif name == 'random':
        return importlib.import_module('.random', package=__name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["evaluation", "creation", "benchmark", "generation", "random"]