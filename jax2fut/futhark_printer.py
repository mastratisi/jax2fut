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
    IfExpr,
    MapExpr,
    Let,
    Function,
    Module
)

import jax
import jax.numpy as jnp
from jax.core import Literal, JaxprEqn, Var as JaxprVar, ClosedJaxpr


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

    result_name = fn.result.var.name

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
            return str(v)

        if isinstance(expr, VarExpr):
            return expr.var.name

        if isinstance(expr, UnaryOp):
            return f"{expr.x.type.base}.{expr.op}({FutharkPrinter.print_expr(expr.x)})"

        if isinstance(expr, BinaryOp):
            return f"({expr.x.type.base}.{expr.op}) {FutharkPrinter.print_expr(expr.x)} {FutharkPrinter.print_expr(expr.y)}"

        if isinstance(expr, ArrayIndex):
            indices = ", ".join(FutharkPrinter.print_expr(i) for i in expr.indices)
            return f"{FutharkPrinter.print_expr(expr.array)}[{indices}]"

        
        if isinstance(expr, IfExpr):
            return (
                f"if {FutharkPrinter.print_expr(expr.cond)}\n"
                f"{ind}  then {FutharkPrinter.print_expr(expr.true_branch, indent + 2)}\n"
                f"{ind}  else {FutharkPrinter.print_expr(expr.false_branch, indent + 2)}"
            )
        if isinstance(expr, MapExpr):
            lambda_args = ",".join([FutharkPrinter.print_var(p) for p in expr.func.params])
            lambda_body = "\n".join(FutharkPrinter.print_let(let, 0) for let in expr.func.body)
            lambdaf = "\\"+lambda_args + " -> " + lambda_body + " " + FutharkPrinter.print_expr(expr.func.result)
            ts =" " + " ".join([FutharkPrinter.print_expr(p) for p in expr.inputs])
            mapN = f"map{'' if len(expr.inputs) == 1 else str(len(expr.inputs))}"
            return (
                f"{mapN} ({lambdaf}) {ts}"
            )

        raise NotImplementedError(f"Expression type: {type(expr)}")

    @staticmethod
    def print_let(let: Let, indent: int = 0) -> str:
        """Convert a Let to Futhark let binding syntax."""
        ind = " " * indent
        return f"{ind}let {FutharkPrinter.print_var(let.var)} = {FutharkPrinter.print_expr(let.expr)} in"

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
            f"{ind} {result}"
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
    """Convenience function to  any Futhark AST node."""
    if isinstance(ast, Module):
        return FutharkPrinter.print_module(ast)
    if isinstance(ast, Function):
        return FutharkPrinter.print_function(ast)
    if isinstance(ast, Let):
        return FutharkPrinter.print_let(ast)
    if isinstance(ast, Expr):
        return FutharkPrinter.print_expr(ast)
    #if isinstance(ast, Var):
    #    return FutharkPrinter.print_var(ast)
    if isinstance(ast, FutharkType):
        return FutharkPrinter.print_type(ast)
        
    raise TypeError(f"Unsupported AST node type: {type(ast)}")
