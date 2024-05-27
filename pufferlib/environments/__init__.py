from pdb import set_trace as T
import pufferlib

def try_import(module_path, package_name=None):
    if package_name is None:
        package_name = module_path
    try:
       package = __import__(module_path)
    except ImportError as e:
        raise ImportError(
            f'{e.args[0]}\n\n'
            'This is probably an installation error. Try: '
            f'pip install pufferlib[{package_name}]. '

            'Note that some environments have non-python dependencies. '
            'These are included in PufferTank. Or, you can install '
            'manually by following the instructions provided by the '
            'environment meaintainers. But some are finicky, so we '
            'recommend using PufferTank.'
        ) from e
    return package
