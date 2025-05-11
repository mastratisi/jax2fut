from typing import List, Any, Dict, Union
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.core import Literal, JaxprEqn, Var as JaxprVar, ClosedJaxpr

# === 1) Define a minimal Futhark AST ===


@dataclass
class FutharkType:
    base: str  # e.g. "f32"
    dims: List[int]  # e.g. [5] for [5]f32

    def __str__(self):
        return f"{''.join(f'[{d}]' for d in self.dims)}{self.base}"


@dataclass
class Var:
    name: str
    type: FutharkType


class Expr:
    pass


@dataclass
class LiteralExpr(Expr):
    value: Any
    type: FutharkType


@dataclass
class UnaryOp(Expr):
    op: str  # "sin", "exp", ...
    x: Expr


@dataclass
class BinaryOp(Expr):
    op: str  # "+", "*", etc.
    x: Expr
    y: Expr


@dataclass
class Let:
    var: Var
    expr: Expr


@dataclass
class Function:
    name: str
    params: List[Var]
    body: List[Let]
    result: Var


# === 2) Helpers: map JAX dtypes & avals to Futhark types ===


def jax_dtype_to_futhark(dtype) -> str:
    if dtype == jnp.float32:
        return "f32"
    if dtype == jnp.float64:
        return "f64"
    if dtype == jnp.int32:
        return "i32"
    if dtype == jnp.int64:
        return "i64"
    raise NotImplementedError(f"dtype {dtype}")


def aval_to_futhark_type(aval) -> FutharkType:
    base = jax_dtype_to_futhark(aval.dtype)
    dims = list(aval.shape)
    return FutharkType(base, dims)


# === 3) The main translator: ClosedJaxpr → Futhark AST ===


def jaxpr_to_futhark_ast(cj: ClosedJaxpr, name: str = "f") -> Function:
    jaxpr = cj.jaxpr
    env: Dict[JaxprVar, Union[Var, LiteralExpr]] = {}
    lets: List[Let] = []

    # 3.1) parameters
    params: List[Var] = []
    for i, v in enumerate(jaxpr.invars):
        vt = aval_to_futhark_type(v.aval)
        fv = Var(f"temp{i}", vt)
        env[v] = fv
        params.append(fv)

    # 3.2) walk equations
    for eqn in jaxpr.eqns:
        # resolve inputs
        inputs: List[Expr] = []
        for iv in eqn.invars:
            if isinstance(iv, Literal):
                lit_t = aval_to_futhark_type(iv.aval)
                inputs.append(LiteralExpr(iv.val, lit_t))
            else:
                inputs.append(env[iv])

        # pick the right AST node
        prim = eqn.primitive.name
        if prim == "add":
            expr = BinaryOp("+", inputs[0], inputs[1])
        elif prim == "mul":
            expr = BinaryOp("*", inputs[0], inputs[1])
        elif prim == "sin":
            expr = UnaryOp("sin", inputs[0])
        # … you can add more primitives here …
        else:
            raise NotImplementedError(f"primitive {prim}")

        # bind the output var
        outv = eqn.outvars[0]
        vt = aval_to_futhark_type(outv.aval)
        fv = Var(f"temp{len(env)}", vt)
        env[outv] = fv
        lets.append(Let(fv, expr))

    # 3.3) final result
    result = env[jaxpr.outvars[0]]
    return Function(name, params, lets, result)


# === 4) Pretty-printer for the AST ===


def print_expr(e: Expr) -> str:
    if isinstance(e, LiteralExpr):
        # format floats with .0
        v = e.value
        if isinstance(v, float) and "." not in repr(v):
            v = f"{v:.1f}"
        return repr(v)
    if isinstance(e, Var):
        return e.name
    if isinstance(e, UnaryOp):
        return f"{e.x.type.base}.{e.op}({print_expr(e.x)})"
    if isinstance(e, BinaryOp):
        assert e.x.type == e.y.type
        return f"({e.x.type.base}.{e.op}) {print_expr(e.x)} {print_expr(e.y)}"
    raise NotImplementedError(f"print_expr: {e}")


def print_function(fn: Function) -> str:
    ps = []
    for p in fn.params:
        # Use proper Futhark type syntax
        type_str = str(p.type)
        ps.append(f"{p.name}: {type_str}")

    body = []
    for let in fn.body:
        orig_expr = print_expr(let.expr)
        body.append(f"let {let.var.name} = {orig_expr}")

    result_name = fn.result.name

    return (
        f"let {fn.name} ({', '.join(ps)}) =\n"
        + "  "
        + "\n  ".join(body)
        + "\n"
        + f"  in {result_name}"
    )
