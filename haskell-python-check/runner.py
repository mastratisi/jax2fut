from ctypes import *
import ctypes
from typing import Any, Callable, Dict, Tuple, Union

# Type aliases for better readability
PythonValue = Union[int, float, str]
TestCase = Tuple[PythonValue, PythonValue]
FunctionSpec = Tuple[Any, list, TestCase]


class HaskellBridge:
    def __init__(self, lib_path: str = "./Test.so"):
        """Initialize the bridge between Python and Haskell."""
        self.lib = cdll.LoadLibrary(lib_path)
        self.lib.hs_init(None, None)
        self.lib.hs_init_python()
        self._setup_functions()

    def _setup_functions(self) -> None:
        """Set up all Haskell functions with their proper types."""
        self.funcs: Dict[str, FunctionSpec] = {
            "addFortyTwo": (c_int, [c_int], (10, 52)),
            "addTenFloat": (c_float, [c_float], (10.0, 20.0)),
            "addTenFloatWithAdd": (c_float, [c_float], (11.0, 21.0)),
            "appendWorld": (c_char_p, [c_char_p], ("hello", "hello world!")),
        }

        # Set up the Python object attribute checker
        self.check_attr = self.lib.checkPythonAttr
        self.check_attr.restype = c_int
        self.check_attr.argtypes = [c_void_p, c_char_p]

    def _to_c_string(self, s: str) -> c_char_p:
        """Convert a Python string to a C string."""
        return c_char_p(s.encode("utf-8"))

    def _from_c_string(self, c_str: c_char_p) -> str:
        """Convert a C string back to a Python string."""
        return string_at(c_str).decode("utf-8")

    def _run_test(self, func_name: str, func_spec: FunctionSpec) -> None:
        """Run a single test case for a Haskell function."""
        f = getattr(self.lib, func_name)
        f.restype, f.argtypes, (input_val, expected) = func_spec

        print(f"{func_name}({input_val}) == {expected}")

        try:
            if func_name == "appendWorld":
                input_c = self._to_c_string(input_val)
                result = f(input_c)
                result_str = self._from_c_string(result)
                if result_str != expected:
                    print(
                        f"Failed: {func_name}({input_val}) != {expected}, got {result_str}"
                    )
            else:
                result = f(input_val)
                if result != expected:
                    print(
                        f"Failed: {func_name}({input_val}) != {expected}, got {result}"
                    )
        except ctypes.ArgumentError as e:
            print(f"Failed: {func_name}({input_val}) raised {e}")

    def check_python_attr(self, obj: Any, attr_name: str) -> bool:
        """Check if a Python object has a specific attribute."""
        obj_ptr = cast(id(obj), c_void_p)
        attr_name_bytes = attr_name.encode("utf-8")
        result = self.check_attr(obj_ptr, attr_name_bytes)
        return bool(result)

    def run_all_tests(self) -> None:
        """Run all test cases."""
        # Run basic function tests
        for func_name, func_spec in self.funcs.items():
            self._run_test(func_name, func_spec)

        # Run Python object attribute tests
        print("\nTesting Python object attribute checking:")
        test_obj = TestObject(has_attr=True)
        empty_obj = TestObject(has_attr=False)

        print(f"Object with test_attr: {self.check_python_attr(test_obj, 'test_attr')}")
        print(
            f"Object without test_attr: {self.check_python_attr(empty_obj, 'test_attr')}"
        )
        print(
            f"Object with some_other_attr: {self.check_python_attr(test_obj, 'some_other_attr')}"
        )

    def cleanup(self) -> None:
        """Clean up Haskell and Python resources."""
        # self.lib.hs_cleanup_python()
        self.lib.hs_exit()


class TestObject:
    """Test class for Python object attribute checking."""

    def __init__(self, has_attr: bool = False):
        if has_attr:
            self.test_attr = True
        self.some_other_attr = "hello"


def main():
    """Main entry point."""
    bridge = HaskellBridge()
    try:
        bridge.run_all_tests()
    finally:
        bridge.cleanup()


if __name__ == "__main__":
    main()
