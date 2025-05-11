# shell.nix
{ pkgs ? import <nixpkgs> { config.cudaSupport = true; config.allowUnfree = true; } }:
let my-python = pkgs.python3;
    python-with-my-packages = my-python.withPackages (p: with p; [
      ipython
      jupyter
      jupyterlab-lsp jedi-language-server python-lsp-server
      sympy


      jax
      jaxlib
      #flax
      
      matplotlib
      numpy
      
#    coconut
    ]);
    haskell = pkgs.haskellPackages.ghcWithPackages (pkgs: with pkgs; [
      haskell-language-server
      
    ]);in
pkgs.mkShell {
  buildInputs = [
    python-with-my-packages
    pkgs.jupyter
    pkgs.gdb
    pkgs.futhark
    haskell
    pkgs.cabal-install
  ];
  shellHook = ''
export PYTHONPATH=${python-with-my-packages}/${python-with-my-packages.sitePackages}
    '';
}
