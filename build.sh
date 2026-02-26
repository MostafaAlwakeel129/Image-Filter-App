#!/bin/bash

# 1. Detect the active Python path
PYTHON_EXE=$(which python)

# 2. Clean up old build artifacts to prevent "sticky" cache issues
echo "--- Cleaning old build files ---"
rm -rf build
rm -f frontend/cv_backend*.so

# 3. Create build directory
mkdir build
cd build

# 4. Run CMake pointing specifically to our active Python
# We pass the executable path so pybind11 finds the matching headers/libs
echo "--- Configuring CMake ---"
cmake .. -DPYTHON_EXECUTABLE=$PYTHON_EXE

# 5. Compile the module using all available CPU cores
echo "--- Compiling ---"
make

# 6. Move the resulting .so file to the frontend directory
# Note: pybind11 often adds a suffix like .cpython-312-x86_64-linux-gnu.so
# We'll rename it to a simple cv_backend.so for your imports.
echo "--- Installing module to frontend/ ---"
cp cv_backend*.so ../frontend/cv_backend.so

echo "--- Done! You can now run your app. ---"
