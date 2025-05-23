from ctypes import *
import ctypes

# Initialize Haskell RTS
lib = cdll.LoadLibrary("./Test.so")
lib.hs_init(None, None)

# Initialize Python in Haskell
lib.hs_init_python()


# Define a test class
class TestObject:
    def __init__(self, has_attr=False):
        if has_attr:
            self.test_attr = True
        self.some_other_attr = "hello"


try:
    # Test the original functions
    funcs = {
        #    name                restype    argtypes    input  expected value
        "addFortyTwo": (c_int, [c_int], (10, 52)),
        "addTenFloat": (c_float, [c_float], (10.0, 20.0)),
        "addTenFloatWithAdd": (c_float, [c_float], (11.0, 21.0)),
        "appendWorld": (c_char_p, [c_char_p], ("hello", "hello world!")),
    }

    for func in funcs:
        f = getattr(lib, func)
        f.restype, f.argtypes, test = funcs[func]
        input, expected = test

        # Special handling for string arguments
        if func == "appendWorld":
            input_bytes = input.encode("utf-8")
            input_c = c_char_p(input_bytes)
            expected_bytes = expected.encode("utf-8")
            expected_c = c_char_p(expected_bytes)
            print("{0}({1}) == {2}".format(func, input, expected))
            try:
                result = f(input_c)
                result_str = string_at(result).decode("utf-8")
                if result_str != expected:
                    print(f"Failed: {func}({input}) != {expected}, got {result_str}")
            except ctypes.ArgumentError as e:
                print(f"Failed: {func}({input}) raised {e}")
        else:
            print("{0}({1}) == {2}".format(func, input, expected))
            try:
                result = f(input)
                if result != expected:
                    print(f"Failed: {func}({input}) != {expected}, got {result}")
            except ctypes.ArgumentError as e:
                print(f"Failed: {func}({input}) raised {e}")

    # Test Python object attribute checking
    print("\nTesting Python object attribute checking:")

    # Set up the checkPythonAttr function
    check_attr = lib.checkPythonAttr
    check_attr.restype = c_int
    check_attr.argtypes = [c_void_p, c_char_p]

    # Create test objects
    obj_with_attr = TestObject(has_attr=True)
    obj_without_attr = TestObject(has_attr=False)

    # Get the Python object's memory address and increment its reference count
    obj_with_attr_ptr = cast(id(obj_with_attr), c_void_p)
    obj_without_attr_ptr = cast(id(obj_without_attr), c_void_p)

    # Test checking for the attribute
    attr_name = b"test_attr"
    other_attr_name = b"some_other_attr"

    # Test object with attribute
    result = check_attr(obj_with_attr_ptr, attr_name)
    print(f"Object with test_attr: {bool(result)}")

    # Test object without attribute
    result = check_attr(obj_without_attr_ptr, attr_name)
    print(f"Object without test_attr: {bool(result)}")

    # Test other attribute that exists
    result = check_attr(obj_with_attr_ptr, other_attr_name)
    print(f"Object with some_other_attr: {bool(result)}")

finally:
    # Cleanup Python in Haskell
    lib.hs_cleanup_python()
    # Cleanup Haskell RTS
    lib.hs_exit()
