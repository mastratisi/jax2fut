from typing import List, Any, Dict, Union, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# === Base Types ===


@dataclass
class FutharkType:
    """Represents a Futhark type with base type and dimensions."""

    base: str  # e.g. "f32", "i32", "bool"
    dims: List[int]  # e.g. [5] for [5]f32
    is_mutable: bool = False  # For mutable arrays

    def __str__(self) -> str:
        """Convert to Futhark type syntax."""
        dims_str = "".join(f"[{d}]" for d in self.dims)
        mut_str = "*" if self.is_mutable else ""
        return f"{dims_str}{mut_str}{self.base}"

    def __repr__(self) -> str:
        return f"FutharkType(base='{self.base}', dims={self.dims}, is_mutable={self.is_mutable})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FutharkType):
            return False
        return (
            self.base == other.base
            and self.dims == other.dims
            and self.is_mutable == other.is_mutable
        )


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
    type: FutharkType

    def __repr__(self) -> str:
        return f"LiteralExpr(value={repr(self.value)}, type={repr(self.type)})"


@dataclass
class VarExpr(Expr):
    """Represents a variable reference."""

    var: Var

    def __init__(self, var: Var) -> None:
        assert isinstance(var, Var), f"Expected Var, got {type(var).__name__}"
        self.var = var

    def __repr__(self) -> str:
        return f"VarExpr(var={repr(self.var)})"


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
class ArrayOp(Expr):
    """Represents array operations like map, reduce, scan."""

    op: str  # e.g. "map", "reduce", "scan"
    f: "Function"  # Function to apply
    array: Expr
    type: FutharkType
    neutral: Optional[Expr] = None  # For reduce/scan

    def __repr__(self) -> str:
        return f"ArrayOp(op='{self.op}', f={repr(self.f)}, array={repr(self.array)}, type={repr(self.type)})"


@dataclass
class IfExpr(Expr):
    """Represents a conditional expression."""

    cond: Expr
    true_branch: Expr
    false_branch: Expr
    type: FutharkType  # Both branches must have same type

    def __repr__(self) -> str:
        return f"IfExpr(cond={repr(self.cond)}, true_branch={repr(self.true_branch)}, false_branch={repr(self.false_branch)}, type={repr(self.type)})"


# === Statement Nodes ===


class Stmt(ABC):
    """Base class for all Futhark statements."""

    @abstractmethod
    def __repr__(self) -> str:
        pass


@dataclass
class Let(Stmt):
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
    body: List[Stmt]
    result: Expr
    type_params: List[str] = []  # For polymorphic functions

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
