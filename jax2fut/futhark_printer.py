from typing import List, Any, Dict, Union
from dataclasses import dataclass
from .futhark_ast import (
    FutharkType,
    Var,
    Expr,
    LiteralExpr,
    VarExpr,
    UnaryOp,
    BinaryOp,
    ArrayIndex,
    ArrayOp,
    IfExpr,
    Let,
    Function,
    Module,
    Stmt,
)

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


class FutharkPrinter:
    """Pretty printer for Futhark AST nodes."""

    @staticmethod
    def print_type(type_: FutharkType) -> str:
        """Convert a FutharkType to Futhark type syntax."""
        return str(type_)

    @staticmethod
    def print_var(var: Var) -> str:
        """Convert a Var to Futhark variable syntax."""
        return f"{var.name}: {FutharkPrinter.print_type(var.type)}"

    @staticmethod
    def print_expr(expr: Expr, indent: int = 0) -> str:
        """Convert an Expr to Futhark expression syntax."""
        ind = " " * indent

        if isinstance(expr, LiteralExpr):
            # Format floats with .0
            v = expr.value
            if isinstance(v, float) and "." not in repr(v):
                v = f"{v:.1f}"
            return repr(v)

        if isinstance(expr, Var):
            return expr.name

        if isinstance(expr, UnaryOp):
            return f"{expr.x.type.base}.{expr.op}({FutharkPrinter.print_expr(expr.x)})"

        if isinstance(expr, BinaryOp):
            return f"({FutharkPrinter.print_expr(expr.x)}) {expr.op} ({FutharkPrinter.print_expr(expr.y)})"

        if isinstance(expr, ArrayIndex):
            indices = ", ".join(FutharkPrinter.print_expr(i) for i in expr.indices)
            return f"{FutharkPrinter.print_expr(expr.array)}[{indices}]"

        if isinstance(expr, ArrayOp):
            if expr.op == "map":
                return f"map {FutharkPrinter.print_function(expr.f)} {FutharkPrinter.print_expr(expr.array)}"
            elif expr.op == "reduce":
                return (
                    f"reduce {FutharkPrinter.print_function(expr.f)} "
                    f"{FutharkPrinter.print_expr(expr.neutral)} "
                    f"{FutharkPrinter.print_expr(expr.array)}"
                )
            elif expr.op == "scan":
                return (
                    f"scan {FutharkPrinter.print_function(expr.f)} "
                    f"{FutharkPrinter.print_expr(expr.neutral)} "
                    f"{FutharkPrinter.print_expr(expr.array)}"
                )
            else:
                raise NotImplementedError(f"Array operation: {expr.op}")

        if isinstance(expr, IfExpr):
            return (
                f"if {FutharkPrinter.print_expr(expr.cond)}\n"
                f"{ind}  then {FutharkPrinter.print_expr(expr.true_branch, indent + 2)}\n"
                f"{ind}  else {FutharkPrinter.print_expr(expr.false_branch, indent + 2)}"
            )

        raise NotImplementedError(f"Expression type: {type(expr)}")

    @staticmethod
    def print_let(let: Let, indent: int = 0) -> str:
        """Convert a Let to Futhark let binding syntax."""
        ind = " " * indent
        return f"{ind}let {FutharkPrinter.print_var(let.var)} = {FutharkPrinter.print_expr(let.expr)}"

    @staticmethod
    def print_function(func: Function, indent: int = 0) -> str:
        """Convert a Function to Futhark function syntax."""
        ind = " " * indent

        # Type parameters
        type_params = ""
        if func.type_params:
            type_params = f" [{', '.join(func.type_params)}]"

        # Parameters
        params = ", ".join(FutharkPrinter.print_var(p) for p in func.params)

        # Body
        body = "\n".join(FutharkPrinter.print_let(let, indent + 2) for let in func.body)

        # Result
        result = FutharkPrinter.print_expr(func.result)

        return (
            f"{ind}let {func.name}{type_params} ({params}) =\n"
            f"{body}\n"
            f"{ind}  in {result}"
        )

    @staticmethod
    def print_module(module: Module) -> str:
        """Convert a Module to Futhark module syntax."""
        # Print all functions
        functions = "\n\n".join(
            FutharkPrinter.print_function(f) for f in module.functions
        )

        # Add entry point annotations
        entry_points = "\n".join(f"entry {name}" for name in module.entry_points)

        return f"{functions}\n\n{entry_points}"


def print_futhark(ast: Any) -> str:
    """Convenience function to print any Futhark AST node."""
    if isinstance(ast, Module):
        return FutharkPrinter.print_module(ast)
    if isinstance(ast, Function):
        return FutharkPrinter.print_function(ast)
    if isinstance(ast, Let):
        return FutharkPrinter.print_let(ast)
    if isinstance(ast, Expr):
        return FutharkPrinter.print_expr(ast)
    if isinstance(ast, Var):
        return FutharkPrinter.print_var(ast)
    if isinstance(ast, FutharkType):
        return FutharkPrinter.print_type(ast)
    raise TypeError(f"Unsupported AST node type: {type(ast)}")
