import futhark_printer as fp
import jaxpr_translator as tranlator

import jax
import time
import jax.numpy as jnp
import subprocess
import numpy as np

jax.config.update("jax_enable_x64", True)


def f(x, y):
    return x + y


x0 = jnp.ones((), jnp.int32)
y0 = jnp.ones((), jnp.int32)
cj = jax.make_jaxpr(f)(x0, y0)

args = [x0, y0]

# Call function to get the JAX output:
JAX_output = f(x0, y0)

# Then generate Futhark code
fn = tranlator.jaxpr_to_futhark(cj, name="main")
src = fp.futhark_codegen(fn)

# Then write that a temp file


folder = "__codegen__"

subprocess.run(["rm", "-f", f"./{folder}/*"])

# Create __codegen_ directory if it doesn't exist
subprocess.run(["mkdir", "-p", folder], check=True)

filename = f"test_{time.time()}"
path = f"{folder}/{filename}"
with open(path + ".fut", "w") as f:
    f.write(src)


subprocess.run(["futhark", "c", path + ".fut"], check=True)
result = subprocess.run(
    ["./" + path],
    input=" ".join(map(str, args)),
    text=True,
    capture_output=True,
)

print("jax output:", JAX_output)
print("futhark output:", result.stdout[:-1])


evalled = f"{result.stdout[:-1]} == {JAX_output}"
print("EVAL STRING:", evalled)


res = subprocess.run(
    ["futhark", "eval", evalled],
    check=True,
    capture_output=True,
)
out = res.stdout
print("hej", out)

assert (
    out == b"true\n"
), f"Outputs do not match: Futhark output = {result.stdout}, JAX output = {JAX_output}, error: {res.stderr}"
