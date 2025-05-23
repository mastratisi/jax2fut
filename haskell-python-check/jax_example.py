import jax
from jax import numpy as jnp
from jax._src.core import JaxprEqn, Jaxpr, Var, Atom
from typing import Any, List, Tuple
import ctypes
from ctypes import c_void_p, c_int, cast, cdll
import os


class HaskellBridge:
    def __init__(self, lib_path: str = "./haskell_interface.so"):
        """Initialize the bridge between Python and Haskell."""
        self.lib = cdll.LoadLibrary(lib_path)
        self.lib.hs_init(None, None)
        self.lib.hs_init_python()
        self._setup_functions()

    def _setup_functions(self) -> None:
        """Set up all Haskell functions with their proper types."""
        self.traverse_jaxpr = self.lib.traverseJaxpr
        self.traverse_jaxpr.restype = c_int
        self.traverse_jaxpr.argtypes = [c_void_p]

    def traverse_jaxpr_tree(self, jaxpr: Jaxpr) -> int:
        """Traverse a JAXPR using Haskell and return the number of nodes."""
        jaxpr_ptr = cast(id(jaxpr), c_void_p)
        return self.traverse_jaxpr(jaxpr_ptr)

    def cleanup(self) -> None:
        """Clean up Haskell resources only (not Python)."""
        # self.lib.hs_cleanup_python()  # Do NOT call this to avoid segfault
        self.lib.hs_exit()


def simple_function(x: jnp.ndarray) -> jnp.ndarray:
    """A simple JAX function that does some basic operations."""
    y = x * 2
    z = y + 1
    return jnp.sin(z)


def main():
    # Create a JAXPR from our function
    x = jnp.array([1.0, 2.0, 3.0])
    jaxpr = jax.make_jaxpr(simple_function)(x)

    # Print the JAXPR for debugging
    print("JAXPR:")
    print(jaxpr)

    # Initialize Haskell bridge
    bridge = HaskellBridge()
    try:
        # Traverse the JAXPR using Haskell
        num_nodes = bridge.traverse_jaxpr_tree(jaxpr.jaxpr)
        print(f"\nHaskell traversal found {num_nodes} nodes")
    finally:
        bridge.cleanup()


if __name__ == "__main__":
    main()
