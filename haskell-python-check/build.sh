#!/bin/bash

PYTHON_INCLUDES=$(python3-config --includes)
PYTHON_LDFLAGS=$(python3-config --ldflags)

# Debug output
python3 --version
python3-config --prefix
python3-config --includes
python3-config --ldflags

# Create the haskell_transpiler_src directory if it doesn't exist
mkdir -p haskell_transpiler_src/HaskellTranspiler

# Build the shared library
ghc -O2 --make \
    -no-hs-main \
    -optl '-shared' \
    -optc '-DMODULE=HaskellInterface' \
    $PYTHON_INCLUDES \
    $PYTHON_LDFLAGS \
    -i. \
    -ihaskell_transpiler_src \
    -o haskell_interface.so haskell_interface.hs

# Clean up
rm -f *.hi *.h *.o HaskellInterface_stub.c
rm -f haskell_transpiler_src/*.hi haskell_transpiler_src/*.h haskell_transpiler_src/*.o
rm -f haskell_transpiler_src/HaskellTranspiler/*.hi haskell_transpiler_src/HaskellTranspiler/*.h haskell_transpiler_src/HaskellTranspiler/*.o