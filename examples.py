"""Examples of using the JAX to Futhark translator.

This module demonstrates how to use the translator with various JAX operations,
from simple arithmetic to more complex array operations.
"""

import jax
from jax.typing import ArrayLike 
import jax.numpy as jnp
from jax2fut import jaxpr_to_futhark, jaxprs_to_futhark_module, print_futhark


def example_simple_arithmetic():
    """Example of translating a simple arithmetic function."""

    def f(x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return jnp.sin(x) + jnp.cos(y) * 2.0

    # Get the jaxpr
    jaxpr = jax.make_jaxpr(f)(1.0, 2.0)
    print(jaxpr)

    # Translate to Futhark
    futhark_ast = jaxpr_to_futhark(jaxpr, name="arithmetic_example")

    # Print the Futhark code
    print("=== Simple Arithmetic Example ===")
    print(print_futhark(futhark_ast))
    print()


def example_array_operations():
    """Example of translating array operations."""

    def f(x: jnp.ndarray) -> jnp.ndarray:
        # Array operations: map, reduce, and element-wise operations
        squared = 2.0 ** x
        summed = jnp.sum(squared)
        return jnp.sqrt(summed)

    # Get the jaxpr with a sample input
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    jaxpr = jax.make_jaxpr(f)(x)
    print(jaxpr)

    # Translate to Futhark
    futhark_ast = jaxpr_to_futhark(jaxpr, name="array_ops_example")

    # Print the Futhark code
    print("=== Array Operations Example ===")
    print(print_futhark(futhark_ast))
    print()


def example_matrix_operations():
    """Example of translating matrix operations."""

    def f(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # Matrix multiplication and element-wise operations
        z = jnp.matmul(x, y)
        return jnp.tanh(z)

    # Get the jaxpr with sample inputs
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    jaxpr = jax.make_jaxpr(f)(x, y)

    # Translate to Futhark
    futhark_ast = jaxpr_to_futhark(jaxpr, name="matrix_ops_example")

    # Print the Futhark code
    print("=== Matrix Operations Example ===")
    print(print_futhark(futhark_ast))
    print()


def example_multiple_functions():
    """Example of translating multiple functions into a module."""

    def f(x: ArrayLike)  -> ArrayLike:
        return jnp.sin(x)

    def g(x: ArrayLike) -> ArrayLike:
        return jnp.cos(x)

    def h(x: ArrayLike) -> ArrayLike:
        return f(x) + g(x)

    # Get jaxprs for all functions
    jaxpr_f = jax.make_jaxpr(f)(1.0)
    jaxpr_g = jax.make_jaxpr(g)(1.0)
    jaxpr_h = jax.make_jaxpr(h)(1.0)

    # Translate to a Futhark module
    functions = [("f", jaxpr_f), ("g", jaxpr_g), ("h", jaxpr_h)]
    entry_points = ["h"]  # Only expose h as an entry point
    module = jaxprs_to_futhark_module(functions, entry_points)

    # Print the Futhark code
    print("=== Multiple Functions Example ===")
    print(print_futhark(module))
    print()


def example_conditional():
    """Example of translating a function with conditionals."""

    def f(x: float, y: float) -> float:
        return jax.lax.cond(x > y, lambda: jnp.sin(x), lambda: jnp.cos(y))

    # Get the jaxpr
    jaxpr = jax.make_jaxpr(f)(1.0, 2.0)

    # Translate to Futhark
    futhark_ast = jaxpr_to_futhark(jaxpr, name="conditional_example")

    # Print the Futhark code
    print("=== Conditional Example ===")
    print(print_futhark(futhark_ast))
    print()


def main():
    """Run all examples."""
    print("Running JAX to Futhark translation examples...\n")

    #example_simple_arithmetic()
    example_array_operations()
    # example_matrix_operations()
    # example_multiple_functions()
    # example_conditional()


if __name__ == "__main__":
    main()
