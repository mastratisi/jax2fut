from typing import List, Any, Dict, Union, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import dataclasses

# === Base Types ===

@dataclass
class TensorType:
    """Represents a Futhark scalar or array type with base type and dimensions."""

    base: str  # e.g. "f32", "i32", "bool"
    dims: List[int]  # e.g. [5] for [5]f32
    is_mutable: bool = False  # For mutable arrays

    def __str__(self) -> str:
        """Convert to Futhark type syntax."""
        dims_str = "".join(f"[{d}]" for d in self.dims)
        mut_str = "*" if self.is_mutable else ""
        return f"{dims_str}{mut_str}{self.base}"

    def __repr__(self) -> str:
        return f"TensorType(base='{self.base}', dims={self.dims}, is_mutable={self.is_mutable})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorType):
            return False
        return (
            self.base == other.base
            and self.dims == other.dims
            and self.is_mutable == other.is_mutable
        )

@dataclass
class FuncType:
    """Represents a Futhark function type with a signature"""
    signature : List["FutharkType"]
    def __str__(self) -> str:
        sigs = [str(sig) for sig in signature]
        return " -> ".join(sigs)

    def __repr__(self) -> str:
        return f"FuncType(signature: {str(self)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FuncType):
            return False
        else:
            self.signature == other.signature
    
FutharkType = TensorType | FuncType

def is_tensor_type(t:FutharkType):
    return isinstance(t, TensorType)

def is_firstorder_type(t:FutharkType):
    if not isinstance(t, FuncType):
        return False
    return all(map(is_tensor, t.signature))
    

def tensorOrWithParans(t:FutharkType) -> str:
    if isinstance(t, TensorType):
        return repr(t)
    else:
        return f"({repr(t)})"

@dataclass
class Var:
    """Represents a Futhark variable with name and type."""

    name: str
    type: FutharkType

    def __repr__(self) -> str:
        return f"Var(name='{self.name}', type={repr(self.type)})"


# === Expression Nodes ===


class Expr(ABC):
    """Base class for all Futhark expressions."""

    type: FutharkType

    @abstractmethod
    def __repr__(self) -> str:
        pass


@dataclass
class LiteralExpr(Expr):
    """Represents a literal value."""

    value: Any
    type: TensorType

    def __repr__(self) -> str:
        return f"LiteralExpr(value={repr(self.value)}, type={repr(self.type)})"


@dataclass
class VarExpr(Expr):
    """Represents a variable reference."""

    var: Var

    @property
    def type(self):
        return self.var.type

    def __init__(self, var: Var) -> None:
        assert isinstance(var, Var), f"Expected Var, got {type(var).__name__}"
        self.var = var

    def __repr__(self) -> str:
        return f"VarExpr(var={repr(self.var)})"

@dataclass
class LetExpr(Expr):
    """Represents a let expression"""
    letVar: Var
    letExpr : Expr
    inExpr : Expr

    def __repr__(self) -> str:
        return f"LetExpr(let {repr(self.letVar)} = {repr(self.letExpr)}, in={repr(self.inExpr)})"
    
@dataclass
class LambdaExpr(Expr):
    type: FuncType = field(init=False)
    params: List[VarExpr]
    body : Expr

    def __post_init__(self):
        param_types = [p.type for p in self.params]
        return_type = self.body.type
        self.type = FuncType(signature=param_types + [return_type])
    
    
    def __repr__(self) -> str:
        str_params = " ".join([tensorOrWithParans(p) for p in self.params])
        str_body = repr(self.body)
        return f"LambdaExpr(params={str_params}, body={str_body})"


@dataclass
class FAppExpr(Expr):
    type: FutharkType
    func: Expr
    args: List[Expr]
    def __repr__(self) -> str:
        str_args = " ".join([repr(p) for p in self.params])
        str_func = repr(self.func)
        return f"LambdaExpr(func={str_func}, params={str_args})"


@dataclass
class UnaryOp(Expr):
    """Represents a unary operation."""

    op: str  # e.g. "sin", "exp", "neg"
    x: Expr
    type: FutharkType  # Same type as input

    def __repr__(self) -> str:
        return f"UnaryOp(op='{self.op}', x={repr(self.x)}, type={repr(self.type)})"


@dataclass
class BinaryOp(Expr):
    """Represents a binary operation."""

    op: str  # e.g. "+", "*", "/", "=="
    x: Expr
    y: Expr
    type: FutharkType  # Result type

    def __repr__(self) -> str:
        return f"BinaryOp(op='{self.op}', x={repr(self.x)}, y={repr(self.y)}, type={repr(self.type)})"


@dataclass
class ArrayIndex(Expr):
    """Represents array indexing."""

    array: Expr
    indices: List[Expr]
    type: FutharkType  # Element type

    def __repr__(self) -> str:
        return f"ArrayIndex(array={repr(self.array)}, indices={repr(self.indices)}, type={repr(self.type)})"


@dataclass
class IfExpr(Expr):
    """Represents a conditional expression."""

    cond: Expr
    true_branch: Expr
    false_branch: Expr
    type: FutharkType  # Both branches must have same type

    def __repr__(self) -> str:
        return f"IfExpr(cond={repr(self.cond)}, true_branch={repr(self.true_branch)}, false_branch={repr(self.false_branch)}, type={repr(self.type)})"


    
@dataclass
class MapExpr(Expr):
    func: "Function"
    inputs: List[VarExpr | LiteralExpr]
    def __repr__(self) -> str:
        return "maplol"


    
# === Statement Nodes ===

@dataclass
class Let:
    """Represents a let binding."""

    var: Var
    expr: Expr

    def __repr__(self) -> str:
        return f"Let(var={repr(self.var)}, expr={repr(self.expr)})"


@dataclass
class Function:
    """Represents a Futhark function definition."""

    name: str
    params: List[Var]
    body: List[Let]
    result: VarExpr
    type_params: List[str] = dataclasses.field(default_factory=list)  # For polymorphic functions

    def __repr__(self) -> str:
        return (
            f"Function(name='{self.name}', params={repr(self.params)}, "
            f"body={repr(self.body)}, result={repr(self.result)}, "
            f"type_params={repr(self.type_params)})"
        )


@dataclass
class Module:
    """Represents a complete Futhark module."""

    functions: List[Function]
    entry_points: List[str]  # Names of entry point functions

    def __repr__(self) -> str:
        return f"Module(functions={repr(self.functions)}, entry_points={repr(self.entry_points)})"
