from typing import List, Dict, Any, Union, Optional, Callable
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.core import Literal, JaxprEqn, Var as JaxprVar, ClosedJaxpr, Primitive, Jaxpr
import enum



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
    Module,
)

# === Helper Types ===

Input = VarExpr | LiteralExpr
Handler = Callable[[List[Input], Dict[str, Any]], Expr]




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




# === utils/helpers for handlers ===
class BroadcastEnum(enum.StrEnum):
    elementwise = "a"
    scalar_matrix = "b"
    matrix_scalar = "c"
    scalar_scalar = "d"
    
def is_elementwise(type1:FutharkType, type2:FutharkType):
    dims1, dims2 = type1.dims, type2.dims
    return dims1 == dims2 and dims1 != []

def is_scalar_matrix(type1:FutharkType, type2:FutharkType):
    dims1, dims2 = type1.dims, type2.dims
    return dims1 != dims2 and dims1 == []

def is_matrix_scalar(type1:FutharkType, type2:FutharkType):
    dims1, dims2 = type1.dims, type2.dims
    return dims1 != dims2 and dims2 == []

def is_scalar_scalar(type1:FutharkType, type2:FutharkType):
    dims1, dims2 = type1.dims, type2.dims
    return dims1 == [] and dims2 == []

def get_broadcast_mode(type1:FutharkType, type2:FutharkType):
    if is_elementwise(type1, type2):
        return BroadcastEnum.elementwise
    elif is_scalar_matrix(type1, type2):
        return BroadcastEnum.scalar_matrix
    elif is_matrix_scalar(type1, type2):
        return BroadcastEnum.matrix_scalar
    elif is_scalar_scalar(type1, type2):
        return BroadcastEnum.scalar_scalar
    else:
        raise ValueError("Incompatible types for broadcasting: dimensions do not match")
    
# === Advance Primitiv Handlers ===

def handle_reduce_sum(inputs: List[Input], params: Dict[str, Any]) -> Expr:
    axes = params["axes"]
    match axes:
        case (0,):
            old_type = inputs[0].var.type
            new_base = old_type.base
            new_dim = old_type.dims[1:]
            return UnaryOp(op="sum", x=inputs[0], type=FutharkType(new_base, new_dim))
    raise Exception("axes shape not implemented in handle_reduce_sum")
    

# === Primitive Translation ===


@dataclass
class PrimitiveHandler:
    """Handler for a specific JAX primitive."""

    name: str
    handler: Handler


class PrimitiveTranslator:
    """Handles translation of JAX primitives to Futhark expressions."""

    def __init__(self):
        self.handlers: Dict[str, PrimitiveHandler] = {}
        self._register_default_handlers()
        self.register_handler("reduce_sum", handle_reduce_sum)
        

    def register_handler(
        self, name: str, handler: Handler
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
            ("pow", "**"),
            ("rem", "%"),
            ("eq", "=="),
            ("ne", "!="),
            ("lt", "<"),
            ("le", "<="),
            ("gt", ">"),
            ("ge", ">="),
        ]:

            def make_binary_handler(op_name: str):
                def handler(inputs: List[Input], params: Dict[str, Any]) -> Expr:
                    assert len(inputs) == 2
                    match (inputs[0].type.dims, inputs[1].type.dims):
                        case ([], []):
                            return BinaryOp(
                                op=op_name,
                                x=inputs[0],
                                y=inputs[1],
                                type=inputs[0].type,  # Result type same as input type
                            )
                            
                        case ([], [h, *t]):
                            print("hej")
                            fvar = Var("fvar", inputs[0].type)
                            lexp = BinaryOp(op=op_name, x=inputs[0], y=VarExpr(fvar), type=inputs[0].type)
                            lvar = Var("lvar", inputs[0].type) 
                            let = Let(lvar, lexp)
                            fexp = Function(name="lambda",
                                            params=[fvar],
                                            body = [let],
                                            result= VarExpr(lvar))
                            return MapExpr(fexp, [inputs[1]])
                    raise Exception("lol")
                             

                    
                    

                return handler

            self.register_handler(op, make_binary_handler(futhark_op))

        # Unary operations
        for op, futhark_op in [("sin", "sin"), ("cos", "cos"),
                               ("exp", "exp"), ("log", "log"),
                               ("neg", "neg"), ("sqrt", "sqrt")]:

            def make_unary_handler(op_name: str):
                def handler(inputs: List[VarExpr], params: Dict[str, Any]) -> Expr:
                    assert len(inputs) == 1
                    return UnaryOp(op=op_name, x=inputs[0], type=inputs[0].type)

                return handler

            self.register_handler(op, make_unary_handler(futhark_op))

    def translate_primitive(
        self, primitive: Primitive, inputs: List[Input], params: Dict[str, Any]
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
        self.env: Dict[JaxprVar, Var] = {}

    def translate_literal(self, lit: Literal) -> LiteralExpr:
        """Translate a JAX literal to a Futhark literal expression."""
        type_ = self.type_translator.aval_to_futhark_type(lit.aval)
        return LiteralExpr(value=lit.val, type=type_)

    def translate_var(self, var: JaxprVar) -> VarExpr:
        """Translate a JAX variable to a Futhark variable or literal."""
        if var not in self.env:
            type_ = self.type_translator.aval_to_futhark_type(var.aval)
            self.env[var] = Var(f"x{len(self.env)}", type_)
        return VarExpr(self.env[var])

    def translate_eqn(self, eqn: JaxprEqn) -> List[Let]:
        """Translate a JAX equation to Futhark let bindings."""
        # Translate inputs
        inputs : List[VarExpr | LiteralExpr]= []
        for invar in eqn.invars:
            if isinstance(invar, Literal):
                inputs.append(self.translate_literal(invar))
            else:
                inputs.append(self.translate_var(invar))

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
        body : List[Let] = []
        for eqn in jaxpr.jaxpr.eqns:
            body.extend(self.expr_translator.translate_eqn(eqn))

        # Get result
        result = self.expr_translator.translate_var(jaxpr.jaxpr.outvars[0])

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
