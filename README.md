# JAX to Futhark Translator

This package provides tools to translate JAX expressions to Futhark code. It preserves types and semantics while generating idiomatic Futhark code.

## Features

- Translates JAX functions to Futhark functions
- Preserves types and semantics
- Supports common JAX operations:
  - Basic arithmetic (+, -, *, /, etc.)
  - Mathematical functions (sin, cos, exp, etc.)
  - Array operations (map, reduce, scan)
  - Matrix operations
  - Conditionals
- Generates proper Futhark syntax
- Modular and extensible design

## Installation

```bash
pip install -e .
```

## Usage

Here's a simple example of translating a JAX function to Futhark:

```python
import jax
import jax.numpy as jnp
from jax2fut import jaxpr_to_futhark, print_futhark

# Define a JAX function
def f(x: float, y: float) -> float:
    return jnp.sin(x) + jnp.cos(y) * 2.0

# Get the jaxpr
jaxpr = jax.make_jaxpr(f)(1.0, 2.0)

# Translate to Futhark
futhark_ast = jaxpr_to_futhark(jaxpr, name="my_function")

# Print the Futhark code
print(print_futhark(futhark_ast))
```

This will generate Futhark code like:

```futhark
let my_function (x0: f32, x1: f32) =
  let x2 = f32.sin(x0)
  let x3 = f32.cos(x1)
  let x4 = (f32.*) x3 2.0
  let x5 = (f32.+) x2 x4
  in x5
```

## Examples

The `examples.py` file contains several examples demonstrating different features:

1. Simple arithmetic operations
2. Array operations (map, reduce, etc.)
3. Matrix operations
4. Multiple functions in a module
5. Conditional expressions

Run the examples with:

```bash
python -m jax2fut.examples
```

## Extending the Translator

The translator is designed to be modular and extensible. You can:

1. Add new primitive handlers in `PrimitiveTranslator`
2. Add new AST nodes in `futhark_ast.py`
3. Extend the printer for new node types
4. Add support for more JAX types

### Adding a New Primitive Handler

```python
from jax2fut import PrimitiveTranslator

translator = PrimitiveTranslator()

def my_handler(inputs: List[Expr], params: Dict[str, Any]) -> Expr:
    # Your translation logic here
    pass

translator.register_handler("my_primitive", my_handler)
```

## Project Structure

- `futhark_ast.py`: Defines the Futhark AST nodes
- `jaxpr_translator.py`: Implements the translation from JAX to Futhark
- `futhark_printer.py`: Pretty prints the AST to Futhark code
- `examples.py`: Example usage of the translator

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 