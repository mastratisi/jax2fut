from typing import List, Dict, Any, Union, Optional, Callable
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.core import Literal, JaxprEqn, Var as JaxprVar, ClosedJaxpr, Primitive, Jaxpr

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

# === Type Translation ===


class TypeTranslator:
    """Handles translation of JAX types to Futhark types."""

    @staticmethod
    def jax_dtype_to_futhark(dtype_str: str) -> str:
        """Convert JAX dtype to Futhark base type."""
        dtype_map = {
            "float32": "f32",
            "float64": "f64",
            "int32": "i32",
            "int64": "i64",
            "bool": "bool",
        }
        if dtype_str not in dtype_map.keys():
            raise NotImplementedError(f"Unsupported JAX dtype: {dtype_str}")
        return dtype_map[dtype_str]

    @staticmethod
    def aval_to_futhark_type(aval) -> FutharkType:
        """Convert JAX abstract value to Futhark type."""
        base = TypeTranslator.jax_dtype_to_futhark(str(aval.dtype))
        dims = list(aval.shape)
        return FutharkType(base=base, dims=dims)


# === Primitive Translation ===


@dataclass
class PrimitiveHandler:
    """Handler for a specific JAX primitive."""

    name: str
    handler: Callable[[List[Expr], Dict[str, Any]], Expr]


class PrimitiveTranslator:
    """Handles translation of JAX primitives to Futhark expressions."""

    def __init__(self):
        self.handlers: Dict[str, PrimitiveHandler] = {}
        self._register_default_handlers()

    def register_handler(
        self, name: str, handler: Callable[[List[Expr], Dict[str, Any]], Expr]
    ):
        """Register a new primitive handler."""
        self.handlers[name] = PrimitiveHandler(name, handler)

    def _register_default_handlers(self):
        """Register default handlers for common primitives."""

        # Binary operations
        for op, futhark_op in [
            ("add", "+"),
            ("sub", "-"),
            ("mul", "*"),
            ("div", "/"),
            ("rem", "%"),
            ("eq", "=="),
            ("ne", "!="),
            ("lt", "<"),
            ("le", "<="),
            ("gt", ">"),
            ("ge", ">="),
        ]:

            def make_binary_handler(op_name: str):
                def handler(inputs: List[Expr], params: Dict[str, Any]) -> Expr:
                    assert len(inputs) == 2
                    return BinaryOp(
                        op=op_name,
                        x=inputs[0],
                        y=inputs[1],
                        type=inputs[0].type,  # Result type same as input type
                    )

                return handler

            self.register_handler(op, make_binary_handler(futhark_op))

        # Unary operations
        for op in ["sin", "cos", "exp", "log", "neg"]:

            def make_unary_handler(op_name: str):
                def handler(inputs: List[Expr], params: Dict[str, Any]) -> Expr:
                    assert len(inputs) == 1
                    return UnaryOp(op=op_name, x=inputs[0], type=inputs[0].type)

                return handler

            self.register_handler(op, make_unary_handler(op))

    def translate_primitive(
        self, primitive: Primitive, inputs: List[Expr], params: Dict[str, Any]
    ) -> Expr:
        """Translate a JAX primitive to a Futhark expression."""
        if primitive.name not in self.handlers:
            raise NotImplementedError(
                f"Translation not implemented for primitive: {primitive.name}"
            )
        return self.handlers[primitive.name].handler(inputs, params)


# === Expression Translation ===


class ExprTranslator:
    """Handles translation of JAX expressions to Futhark expressions."""

    def __init__(self):
        self.type_translator = TypeTranslator()
        self.primitive_translator = PrimitiveTranslator()
        self.env: Dict[JaxprVar, Union[Var, LiteralExpr]] = {}

    def translate_literal(self, lit: Literal) -> LiteralExpr:
        """Translate a JAX literal to a Futhark literal expression."""
        type_ = self.type_translator.aval_to_futhark_type(lit.aval)
        return LiteralExpr(value=lit.val, type=type_)

    def translate_var(self, var: JaxprVar) -> Union[Var, LiteralExpr]:
        """Translate a JAX variable to a Futhark variable or literal."""
        if var not in self.env:
            type_ = self.type_translator.aval_to_futhark_type(var.aval)
            self.env[var] = Var(f"x{len(self.env)}", type_)
        return self.env[var]

    def translate_eqn(self, eqn: JaxprEqn) -> List[Let]:
        """Translate a JAX equation to Futhark let bindings."""
        # Translate inputs
        inputs = []
        for invar in eqn.invars:
            if isinstance(invar, Literal):
                inputs.append(self.translate_literal(invar))
            else:
                inputs.append(VarExpr(self.translate_var(invar)))

        # Translate the primitive
        expr = self.primitive_translator.translate_primitive(
            eqn.primitive, inputs, eqn.params
        )

        # Create let bindings for outputs
        lets = []
        for outvar in eqn.outvars:
            type_ = self.type_translator.aval_to_futhark_type(outvar.aval)
            var = Var(f"x{len(self.env)}", type_)
            self.env[outvar] = var
            lets.append(Let(var=var, expr=expr))

        return lets


# === Function Translation ===


class FunctionTranslator:
    """Handles translation of JAX functions to Futhark functions."""

    def __init__(self):
        self.expr_translator = ExprTranslator()

    def translate_function(self, jaxpr: ClosedJaxpr, name: str = "f") -> Function:
        """Translate a JAX function to a Futhark function."""
        # Reset environment
        self.expr_translator.env.clear()

        # Translate parameters
        params = []
        for i, invar in enumerate(jaxpr.jaxpr.invars):
            type_ = self.expr_translator.type_translator.aval_to_futhark_type(
                invar.aval
            )
            var = Var(f"x{i}", type_)
            self.expr_translator.env[invar] = var
            params.append(var)

        # Translate body
        body = []
        for eqn in jaxpr.jaxpr.eqns:
            body.extend(self.expr_translator.translate_eqn(eqn))

        # Get result
        result = VarExpr(self.expr_translator.translate_var(jaxpr.jaxpr.outvars[0]))

        return Function(name=name, params=params, body=body, result=result)


# === Module Translation ===


class ModuleTranslator:
    """Handles translation of JAX modules to Futhark modules."""

    def __init__(self):
        self.function_translator = FunctionTranslator()

    def translate_module(
        self, functions: List[tuple[str, ClosedJaxpr]], entry_points: List[str]
    ) -> Module:
        """Translate a collection of JAX functions to a Futhark module."""
        futhark_functions = []
        for name, jaxpr in functions:
            futhark_functions.append(
                self.function_translator.translate_function(jaxpr, name)
            )
        return Module(functions=futhark_functions, entry_points=entry_points)


# === Main Translation Function ===


def jaxpr_to_futhark(jaxpr: ClosedJaxpr, name: str = "f") -> Function:
    """Convenience function to translate a single JAX function to Futhark."""
    translator = FunctionTranslator()
    return translator.translate_function(jaxpr, name)


def jaxprs_to_futhark_module(
    functions: List[tuple[str, ClosedJaxpr]], entry_points: List[str]
) -> Module:
    """Convenience function to translate multiple JAX functions to a Futhark module."""
    translator = ModuleTranslator()
    return translator.translate_module(functions, entry_points)
