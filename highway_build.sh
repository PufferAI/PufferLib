python setup.py build_ext --inplace  
mv c_highway.cpython-311-darwin.so pufferlib/environments/ocean/highway/
python pufferlib/environments/ocean/highway/highway.py