import pufferlib
import pufferlib.registry

# Populate registry once instead of per-test
bindings = pufferlib.registry.make_all_bindings()
