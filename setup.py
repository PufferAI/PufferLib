# Debug command:
#    DEBUG=1 python setup.py build_ext --inplace --force
#    CUDA_VISIBLE_DEVICES=None LD_PRELOAD=$(gcc -print-file-name=libasan.so) python3.12 -m pufferlib.clean_pufferl eval --train.device cpu

from setuptools import find_packages, find_namespace_packages, setup, Extension
import numpy
import os
import glob
import urllib.request
import zipfile
import tarfile
import platform
import shutil
import pybind11

from setuptools.command.build_ext import build_ext

# Build flags
DEBUG = os.getenv("DEBUG", "0") == "1"
NO_OCEAN = os.getenv("NO_OCEAN", "0") == "1"
NO_TRAIN = os.getenv("NO_TRAIN", "0") == "1"

# Torch is only needed for profiler builds (build_profile_torch, build_profiler).
# The main _C.so build is always torch-free.
HAS_TORCH = False
if not NO_TRAIN:
    try:
        from torch.utils import cpp_extension
        from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME
        HAS_TORCH = True
    except ImportError:
        pass

# Use ccache if available for faster rebuilds
if shutil.which('ccache'):
    os.environ.setdefault('CC', 'ccache cc')
    os.environ.setdefault('CXX', 'ccache c++')

# Build raylib for your platform
RAYLIB_URL = 'https://github.com/raysan5/raylib/releases/download/5.5/'
RAYLIB_NAME = 'raylib-5.5_macos' if platform.system() == "Darwin" else 'raylib-5.5_linux_amd64'
RLIGHTS_URL = 'https://raw.githubusercontent.com/raysan5/raylib/refs/heads/master/examples/shaders/rlights.h'

def download_raylib(platform, ext):
    if not os.path.exists(platform):
        print(f'Downloading Raylib {platform}')
        urllib.request.urlretrieve(RAYLIB_URL + platform + ext, platform + ext)
        if ext == '.zip':
            with zipfile.ZipFile(platform + ext, 'r') as zip_ref:
                zip_ref.extractall()
        else:
            with tarfile.open(platform + ext, 'r') as tar_ref:
                tar_ref.extractall()

        os.remove(platform + ext)
        urllib.request.urlretrieve(RLIGHTS_URL, platform + '/include/rlights.h')

if not NO_OCEAN:
    download_raylib('raylib-5.5_webassembly', '.zip')
    download_raylib(RAYLIB_NAME, '.tar.gz')

BOX2D_URL = 'https://github.com/capnspacehook/box2d/releases/latest/download/'
BOX2D_NAME = 'box2d-macos-arm64' if platform.system() == "Darwin" else 'box2d-linux-amd64'

def download_box2d(platform):
    if not os.path.exists(platform):
        ext = ".tar.gz"

        print(f'Downloading Box2D {platform}')
        urllib.request.urlretrieve(BOX2D_URL + platform + ext, platform + ext)
        with tarfile.open(platform + ext, 'r') as tar_ref:
            tar_ref.extractall()

        os.remove(platform + ext)

if not NO_OCEAN:
    download_box2d('box2d-web')
    download_box2d(BOX2D_NAME)

# Shared compile args for all platforms
extra_compile_args = [
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
    '-DPLATFORM_DESKTOP',
]
extra_link_args = [
    '-fwrapv',
    '-fopenmp',
]
cxx_args = [
    '-fdiagnostics-color=always',
    '-std=c++17',
    '-fopenmp',
]
nvcc_args = [
    '-Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1',
    '-std=c++17',
]

if DEBUG:
    extra_compile_args += [
        '-O0',
        '-g',
        #'-fsanitize=address,undefined,bounds,pointer-overflow,leak',
        #'-fno-omit-frame-pointer',
    ]
    extra_link_args += [
        '-g',
        #'-fsanitize=address,undefined,bounds,pointer-overflow,leak',
    ]
    cxx_args += [
        '-O0',
        '-g',
    ]
    nvcc_args += [
        '-O0',
        '-g',
        '-fsanitize=address',
        '-fno-omit-frame-pointer',
        '-lineinfo',
    ]
else:
    extra_compile_args += [
        '-O2',
        '-flto=auto',
        '-fno-semantic-interposition',
        '-fvisibility=hidden',
    ]
    extra_link_args += [
        '-O2',
    ]
    cxx_args += [
        '-O2',
        '-fno-semantic-interposition',
        '-Wno-c++11-narrowing',
    ]
    nvcc_args += [
        '-O3',
    ]

system = platform.system()
if system == 'Linux':
    extra_compile_args += [
        '-Wno-alloc-size-larger-than',
        '-fmax-errors=3',
    ]
    extra_link_args += [
        '-Bsymbolic-functions',
    ]
elif system == 'Darwin':
    extra_compile_args += [
        '-Wno-error=int-conversion',
        '-Wno-error=incompatible-function-pointer-types',
        '-Wno-error=implicit-function-declaration',
    ]
    extra_link_args += [
        '-framework', 'Cocoa',
        '-framework', 'OpenGL',
        '-framework', 'IOKit',
    ]
else:
    raise ValueError(f'Unsupported system: {system}')

# Default Gym/Gymnasium/PettingZoo versions
# Gym:
# - 0.26 still has deprecation warnings and is the last version of the package
# - 0.25 adds a breaking API change to reset, step, and render_modes
# - 0.24 is broken
# - 0.22-0.23 triggers deprecation warnings by calling its own functions
# - 0.21 is the most stable version
# - <= 0.20 is missing dict methods for gym.spaces.Dict
# - 0.18-0.21 require setuptools<=65.5.0

# Extensions
extnames = ["pufferlib._C"]

class BuildExt(build_ext):
    user_options = build_ext.user_options + [
        ('precision=', None, 'Precision: float or bf16 (default: bf16)'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.precision = 'bf16'

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        # Propagate any build_ext options (e.g., --inplace, --force) to subcommands
        build_ext_opts = self.distribution.command_options.get('build_ext', {})
        if build_ext_opts:
            self.distribution.command_options['build_c'] = build_ext_opts.copy()

        # Build _C.so (always torch-free)
        if not NO_TRAIN:
            _build_notorch_C(force=self.force, precision=self.precision)
        self.run_command('build_c')

class CBuildExt(build_ext):
    def run(self, *args, **kwargs):
        self.extensions = [e for e in self.extensions if e.name not in extnames]
        super().run(*args, **kwargs)

# Define cmdclass outside of setup to add dynamic commands
cmdclass = {
    "build_ext": BuildExt,
    "build_c": CBuildExt,
}


INCLUDE = [f'{BOX2D_NAME}/include', f'{BOX2D_NAME}/src']
RAYLIB_A = f'{RAYLIB_NAME}/lib/libraylib.a'
extension_kwargs = dict(
    include_dirs=INCLUDE,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    extra_objects=[RAYLIB_A],
)

# Find C extensions
c_extensions = []
c_extension_paths = []
if not NO_OCEAN:
    c_extension_paths = glob.glob('ocean/**/binding.c', recursive=True)
    c_extensions = [
        Extension(
            path.rstrip('.c').replace('/', '.'),
            sources=[path],
            **extension_kwargs,
        )
        for path in c_extension_paths if 'matsci' not in path
    ]
    c_extension_paths = [os.path.join(*path.split('/')[:-1]) for path in c_extension_paths]

    for c_ext in c_extensions:
        if "impulse_wars" in c_ext.name:
            print(f"Adding {c_ext.name} to extra objects")
            c_ext.extra_objects.append(f'{BOX2D_NAME}/libbox2d.a')
            # TODO: Figure out why this is necessary for some users
            impulse_include = 'ocean/impulse_wars/include'
            if impulse_include not in c_ext.include_dirs:
                c_ext.include_dirs.append(impulse_include)

        if 'matsci' in c_ext.name:
            c_ext.include_dirs.append('/usr/local/include')
            c_ext.extra_link_args.extend(['-L/usr/local/lib', '-llammps'])


class ProfileTorchBuildExt(build_ext):
    """Build libprofile_torch.a — static torch library (slow, compile once)."""
    user_options = build_ext.user_options + [
        ('precision=', None, 'Precision: float or bf16 (default: float)'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.precision = 'float'

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        import subprocess
        import sysconfig
        import torch.utils.cpp_extension as cpp_ext

        src = 'profile_torch_lib.cu'
        obj = 'profile_torch_lib.o'
        lib = 'libprofile_torch.a'

        nvcc = cpp_ext._join_cuda_home('bin', 'nvcc')
        arch = '-arch=sm_89'
        lib_paths = cpp_ext.library_paths()
        nvtx_lib_dir = os.path.join(cpp_ext.CUDA_HOME, 'lib64')

        precision_flag = '-DPRECISION_FLOAT' if self.precision == 'float' else ''

        # Step 1: Compile to object file
        cmd = [nvcc, '-c', '-O3', arch, '-DUSE_TORCH', '-DPUFFERLIB_TORCH',
               '-I.', '-Isrc',
               '-Xcompiler', '-fPIC']
        if precision_flag:
            cmd.append(precision_flag)
        cmd += ['-I' + sysconfig.get_path('include')]
        cmd += ['-I' + p for p in cpp_ext.include_paths()]
        cmd += [src, '-o', obj]

        print(f'Building torch lib object: {" ".join(cmd)}')
        subprocess.check_call(cmd)

        # Step 2: Create static library
        ar_cmd = ['ar', 'rcs', lib, obj]
        print(f'Creating static library: {" ".join(ar_cmd)}')
        subprocess.check_call(ar_cmd)
        print(f'Built: {lib}')

cmdclass["build_profile_torch"] = ProfileTorchBuildExt


class ProfilerBuildExt(build_ext):
    """Build profile binary — fast compile, links libprofile_torch.a if present."""
    user_options = build_ext.user_options + [
        ('no-torch', None, 'Build profiler without torch support'),
        ('env=', None, 'Static env to link (e.g., breakout, drive)'),
        ('precision=', None, 'Precision: float or bf16 (default: float)'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.no_torch = False
        self.env = None
        self.precision = 'float'

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        import subprocess
        import sysconfig
        import torch.utils.cpp_extension as cpp_ext

        src = 'profile_kernels.cu'
        out = 'profile'
        torch_lib = 'libprofile_torch.a'

        nvcc = cpp_ext._join_cuda_home('bin', 'nvcc')
        arch = '-arch=sm_89'

        precision_flag = '-DPRECISION_FLOAT' if self.precision == 'float' else ''

        # Check if we should link torch
        use_torch = not self.no_torch and os.path.exists(torch_lib)
        if not self.no_torch and not os.path.exists(torch_lib):
            print(f'Warning: {torch_lib} not found. Build it with: python setup.py build_profile_torch')
            print('Building without torch support.')

        if not use_torch:
            # No-torch build: fast, kernel-only
            cmd = [nvcc, '-O3', arch, '-I.']
            if precision_flag:
                cmd.append(precision_flag)
            cmd += [src, '-o', out]
        else:
            out = 'profile_torch'
            lib_paths = cpp_ext.library_paths()
            nvtx_lib_dir = os.path.join(cpp_ext.CUDA_HOME, 'lib64')

            cmd = [nvcc, '-O3', arch, '-DUSE_TORCH', '-DPUFFERLIB_TORCH',
                   '-I.', '-Isrc']
            if precision_flag:
                cmd.append(precision_flag)

            # Optional static env for envspeed profiling
            if self.env:
                static_lib = f'src/libstatic_{self.env}.a'
                if not os.path.exists(static_lib):
                    raise RuntimeError(f'Static library not found: {static_lib}\n'
                                       f'Build it first with: python setup.py build_{self.env}')
                cmd += [f'-DENV_NAME={self.env}', '-DUSE_STATIC_ENV',
                        f'-I./{RAYLIB_NAME}/include']

            cmd += ['-I' + sysconfig.get_path('include')]
            cmd += ['-I' + p for p in cpp_ext.include_paths()]
            cmd += [src]
            # Link the static torch lib
            cmd += [torch_lib]
            cmd += ['-L' + p for p in lib_paths]
            cmd += ['-L' + nvtx_lib_dir]
            cmd += ['-Xlinker', '-rpath,' + ':'.join(lib_paths)]
            cmd += ['-Xlinker', '--no-as-needed']
            cmd += ['-lc10', '-lc10_cuda', '-ltorch', '-ltorch_cpu', '-ltorch_cuda',
                    '-lcublas', '-lcusolver', '-lcurand',
                    '-lnvToolsExt', '-ldl', '-lnccl']
            cmd += ['-Xlinker', '--unresolved-symbols=ignore-in-shared-libs']
            cmd += ['-Xlinker', '--allow-multiple-definition']
            if self.env:
                static_lib = f'src/libstatic_{self.env}.a'
                cmd += [static_lib, f'./{RAYLIB_NAME}/lib/libraylib.a', '-lGL']
            cmd += ['-lomp5']
            cmd += ['-o', out]

        print(f'Building profiler: {" ".join(cmd)}')
        subprocess.check_call(cmd)
        print(f'Built: {out}')

cmdclass["build_profiler"] = ProfilerBuildExt

# Static env builds: clang-compiled env + gcc/nvcc torch extension
# Discover envs by listing folders in ocean/
OCEAN_SRC_DIR = 'ocean'
STATIC_ENVS = [
    name for name in os.listdir(OCEAN_SRC_DIR)
    if os.path.isdir(os.path.join(OCEAN_SRC_DIR, name))
    and os.path.exists(f'ocean/{name}/binding.c')
]

def _extract_obs_tensor_t(obj_path):
    """Extract OBS_TENSOR_T from a compiled .o via the embedded dtype_symbol string."""
    import subprocess
    out = subprocess.check_output(['strings', obj_path], text=True)
    for line in out.splitlines():
        if line.endswith('Tensor'):
            return line.strip()
    raise RuntimeError(f'Could not find OBS_TENSOR_T in {obj_path}')

def _build_static_lib(env_name, force=False):
    """Build a static .a library for a given env using clang."""
    import subprocess
    env_binding_src = f'ocean/{env_name}/binding.c'
    static_lib = f'src/libstatic_{env_name}.a'
    static_obj = f'src/libstatic_{env_name}.o'

    env_deps = [env_binding_src, 'src/vecenv.h']
    if not force and not _needs_rebuild(static_lib, env_deps):
        print(f'Static env up to date: {static_lib}')
        return static_lib, _extract_obs_tensor_t(static_obj)

    clang_cmd = [
        'clang', '-c', '-O2', '-DNDEBUG',
        '-I.', '-Isrc', f'-Iocean/{env_name}',
        f'-I./{RAYLIB_NAME}/include', '-I/usr/local/cuda/include',
        '-DPLATFORM_DESKTOP',
        '-fno-semantic-interposition', '-fvisibility=hidden',
        '-fPIC', '-fopenmp',
        env_binding_src, '-o', static_obj
    ]
    print(f'Building static env: {" ".join(clang_cmd)}')
    subprocess.check_call(clang_cmd)

    ar_cmd = ['ar', 'rcs', static_lib, static_obj]
    print(f'Creating static library: {" ".join(ar_cmd)}')
    subprocess.check_call(ar_cmd)
    obs_tensor_t = _extract_obs_tensor_t(static_obj)
    print(f'OBS_TENSOR_T={obs_tensor_t}')
    return static_lib, obs_tensor_t

def _needs_rebuild(output, sources):
    """Check if output needs rebuilding based on source mtimes."""
    if not os.path.exists(output):
        return True
    out_mtime = os.path.getmtime(output)
    for src in sources:
        if os.path.exists(src) and os.path.getmtime(src) > out_mtime:
            return True
    return False

# Headers that affect the single compilation unit (for incremental builds)
_BINDINGS_CU_DEPS = [
    'src/bindings.cu', 'src/pufferlib.cu',
    'src/models.cu', 'src/kernels.cu',
    'src/vecenv.h',
    'src/puffernet.h',
]

def _build_notorch_C(static_lib=None, obs_tensor_t=None, force=False, precision='bf16'):
    """Build _C.so via single nvcc compile (no torch dependency)."""
    import subprocess
    import sysconfig

    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or '/usr/local/cuda'
    nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    output = f'pufferlib/_C{ext_suffix}'

    bindings_cu = 'src/bindings.cu'
    bindings_o = 'src/bindings.o'

    precision_flag = '-DPRECISION_FLOAT' if precision == 'float' else ''

    # Check what needs rebuilding
    need_compile = force or _needs_rebuild(bindings_o, _BINDINGS_CU_DEPS)
    need_link = need_compile or force or _needs_rebuild(output, [bindings_o] + ([static_lib] if static_lib else []))

    if not need_link:
        print(f'Up to date: {output}')
        return

    python_include = sysconfig.get_path('include')
    pybind_include = pybind11.get_include()
    nvcc_cmd = [
        nvcc, '-c', '-Xcompiler', '-fPIC',
        '-Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1',
        '-Xcompiler=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-Xcompiler=-DPLATFORM_DESKTOP',
        '-std=c++17',
        '-I.', '-Isrc',
        f'-I{python_include}',
        f'-I{pybind_include}',
        f'-I{numpy.get_include()}',
        f'-I{cuda_home}/include',
        f'-I{RAYLIB_NAME}/include',
        '-Xcompiler=-fopenmp',
    ]
    if precision_flag:
        nvcc_cmd.append(precision_flag)
    if obs_tensor_t:
        nvcc_cmd.append(f'-DOBS_TENSOR_T={obs_tensor_t}')
    if DEBUG:
        nvcc_cmd += ['-O0', '-g']
    else:
        nvcc_cmd += ['-O3']
    nvcc_cmd += [bindings_cu, '-o', bindings_o]

    if need_compile:
        print(f'nvcc: {" ".join(nvcc_cmd)}')
        subprocess.check_call(nvcc_cmd)
    else:
        print(f'nvcc: up to date ({bindings_o})')

    # Link into shared library
    link_cmd = [
        'g++', '-shared', '-fPIC',
        '-fopenmp',
        bindings_o,
    ]
    if static_lib:
        link_cmd.append(static_lib)
    link_cmd += [RAYLIB_A]
    link_cmd += [
        f'-L{cuda_home}/lib64',
        '-lcudart', '-lnccl', '-lnvidia-ml', '-lcublas', '-lcusolver', '-lcurand', '-lcudnn',
        '-lnvToolsExt', '-lomp5',
    ]
    if DEBUG:
        link_cmd += ['-g']
    else:
        link_cmd += ['-O2']
    link_cmd += extra_link_args
    link_cmd += ['-o', output]
    print(f'link: {" ".join(link_cmd)}')
    subprocess.check_call(link_cmd)
    print(f'Built: {output}')

def create_static_env_build_class(env_name):
    """Create a build class that compiles env with clang, then links _C.so."""
    class StaticEnvBuildExt(build_ext):
        user_options = build_ext.user_options + [
            ('precision=', None, 'Precision: float or bf16 (default: bf16)'),
        ]

        def initialize_options(self):
            super().initialize_options()
            self.precision = 'bf16'

        def finalize_options(self):
            super().finalize_options()

        def run(self):
            static_lib, obs_tensor_t = _build_static_lib(env_name, force=self.force)
            _build_notorch_C(static_lib, obs_tensor_t, force=self.force, precision=self.precision)
    return StaticEnvBuildExt

# Add build_<env> for static-linked envs (always available, torch optional)
for env_name in STATIC_ENVS:
    cmdclass[f"build_{env_name}"] = create_static_env_build_class(env_name)

if not NO_OCEAN:
    def create_env_build_class(full_name):
        class EnvBuildExt(build_ext):
            def run(self):
                self.extensions = [e for e in self.extensions if e.name == full_name]
                super().run()
        return EnvBuildExt

    # Add a build_<env>_so command for each env (dynamic .so build)
    for c_ext in c_extensions:
        env_name = c_ext.name.split('.')[-2]
        cmdclass[f"build_{env_name}_so"] = create_env_build_class(c_ext.name)


# No torch extensions in the default build — _C.so is built via subprocess (torch-free).
# Use build_profile_torch / build_profiler for torch-based profiling.
torch_extensions = []

# Prevent Conda from injecting garbage compile flags
from distutils.sysconfig import get_config_vars
cfg_vars = get_config_vars()
for key in ('CC', 'CXX', 'LDSHARED'):
    if cfg_vars[key]:
        cfg_vars[key] = cfg_vars[key].replace('-B /root/anaconda3/compiler_compat', '')
        cfg_vars[key] = cfg_vars[key].replace('-pthread', '')
        cfg_vars[key] = cfg_vars[key].replace('-fno-strict-overflow', '')

for key, value in cfg_vars.items():
    if value and '-fno-strict-overflow' in str(value):
        cfg_vars[key] = value.replace('-fno-strict-overflow', '')

install_requires = [
    'setuptools',
    'numpy<2.0',
    'shimmy[gym-v21]',
    'gym==0.23',
    'gymnasium>=0.29.1',
    'pettingzoo>=1.24.1',
]

if not NO_TRAIN:
    install_requires += [
        'torch>=2.9',
        'psutil',
        'nvidia-ml-py',
        'rich',
        'rich_argparse',
        'imageio',
        'gpytorch',
        'scikit-learn',
        'heavyball>=2.2.0', # contains relevant fixes compared to 1.7.2 and 2.1.1
        'neptune',
        'wandb',
    ]

setup(
    version="3.0.0",
    packages=find_namespace_packages() + find_packages() + c_extension_paths + ['src'],
    package_data={
        "pufferlib": [RAYLIB_NAME + '/lib/libraylib.a']
    },
    include_package_data=True,
    install_requires=install_requires,
    ext_modules = c_extensions,
    cmdclass=cmdclass,
    include_dirs=[numpy.get_include(), RAYLIB_NAME + '/include'],
)
