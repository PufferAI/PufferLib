from pdb import set_trace as T
import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from c_test import player_dtype, test_struct, test_struct_view

dtype = player_dtype()
data = np.zeros((10, 5, dtype.itemsize), dtype=np.uint8)
struct_data = np.frombuffer(data, dtype=dtype).view(np.recarray).reshape(10, 5)
struct_data.x = np.arange(1, 51).reshape(10, 5)

test_struct(struct_data)
test_struct_view()
