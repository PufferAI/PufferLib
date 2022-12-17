import pufferlib

# Populate registry once instead of per-test
bindings = pufferlib.bindings.registry.make_all_bindings()