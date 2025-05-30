"""JAX to Futhark translator.

This package provides tools to translate JAX expressions to Futhark code.
The translation preserves types and semantics while generating idiomatic Futhark code.
"""

from futhark_ast import (
    FutharkType,
    Var,
    Expr,
    LiteralExpr,
    VarExpr,
    UnaryOp,
    BinaryOp,
    ArrayIndex,
    IfExpr,
)
from jaxpr_translator import (
    jaxpr_to_futhark,
    jaxprs_to_futhark_module,
    TypeTranslator,
    PrimitiveTranslator,
    ExprTranslator,
    FunctionTranslator,
    ModuleTranslator,
)
from futhark_printer import futhark_codegen

__all__ = [
    # AST nodes
    "FutharkType",
    "Var",
    "Expr",
    "LiteralExpr",
    "VarExpr",
    "UnaryOp",
    "BinaryOp",
    "ArrayIndex",
    "IfExpr",
    # Translators
    "jaxpr_to_futhark",
    "jaxprs_to_futhark_module",
    "TypeTranslator",
    "PrimitiveTranslator",
    "ExprTranslator",
    "FunctionTranslator",
    "ModuleTranslator",
    # Printer
    "futhark_codegen",
]
