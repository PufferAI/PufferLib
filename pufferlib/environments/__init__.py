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

@pufferlib.dataclass
class EnvArgs:
    pass

@pufferlib.dataclass
class DefaultPolicyArgs:
    input_size: int = 128
    hidden_size: int = 128

@pufferlib.dataclass
class ConvolutionalPolicyArgs:
    framestack: int = 1
    flat_size: int = 7*7*64
    input_size: int = 512
    hidden_size: int = 512
    output_size: int = 512
    channels_last: bool = False
    downsample: int = 1

@pufferlib.dataclass
class RecurrentArgs:
    input_size: int = 128
    hidden_size: int = 128
    num_layers: int = 1
