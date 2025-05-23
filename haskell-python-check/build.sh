#!/bin/bash

PYTHON_INCLUDES=$(python3-config --includes)
PYTHON_LDFLAGS=$(python3-config --ldflags)

# Debug output
python3 --version
python3-config --prefix
python3-config --includes
python3-config --ldflags

ghc -O2 --make \
    -no-hs-main \
    -optl '-shared' \
    -optc '-DMODULE=Test' \
    $PYTHON_INCLUDES \
    $PYTHON_LDFLAGS \
    -o Test.so Test.hs module_init.c

rm -f *.hi *.h *.o Test_stub.c