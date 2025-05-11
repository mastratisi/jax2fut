"""JAX to Futhark translator.

This package provides tools to translate JAX expressions to Futhark code.
The translation preserves types and semantics while generating idiomatic Futhark code.
"""

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
)
from .jaxpr_translator import (
    jaxpr_to_futhark,
    jaxprs_to_futhark_module,
    TypeTranslator,
    PrimitiveTranslator,
    ExprTranslator,
    FunctionTranslator,
    ModuleTranslator,
)
from .futhark_printer import print_futhark

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
    "ArrayOp",
    "IfExpr",
    "Let",
    "Function",
    "Module",
    # Translators
    "jaxpr_to_futhark",
    "jaxprs_to_futhark_module",
    "TypeTranslator",
    "PrimitiveTranslator",
    "ExprTranslator",
    "FunctionTranslator",
    "ModuleTranslator",
    # Printer
    "print_futhark",
]
