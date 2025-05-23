ghc -O2 --make \
      -no-hs-main -optl '-shared' -optc '-DMODULE=Test' \
      -optl '-L/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib/python3.13/config-3.13-darwin' \
      -optl '-lpython3.13' \
      -optl '-ldl' \
      -framework CoreFoundation \
      -o Test.so Test.hs module_init.c

rm *.hi *.h *.o
rm Test_stub.c