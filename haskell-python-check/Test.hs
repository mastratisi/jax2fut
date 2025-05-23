{-# LANGUAGE ForeignFunctionInterface #-}
module Test where
 
import Foreign.C.Types
import Foreign.C.String
import Foreign.Ptr
import Foreign.StablePtr
import Foreign.Marshal.Alloc

-- Import Python C API functions
foreign import ccall "Python.h PyObject_GetAttrString" pyGetAttrString :: Ptr () -> CString -> IO (Ptr ())
foreign import ccall "Python.h PyUnicode_AsUTF8" pyUnicodeAsUTF8 :: Ptr () -> IO CString
foreign import ccall "Python.h PyObject_IsTrue" pyObjectIsTrue :: Ptr () -> IO CInt

add :: Num a => a -> a -> a
add x y = x + y

addFortyTwo :: CInt -> IO CInt
addFortyTwo x = do
    return (42 + x)
 
addTenFloat :: CFloat -> IO CFloat
addTenFloat x = do
    return (10.0 + x)

addTenFloatWithAdd :: CFloat -> IO CFloat
addTenFloatWithAdd x = do
    return (add 10.0 x)
    
appendWorld :: CString -> IO CString
appendWorld s = do
    w <- peekCString s
    newCString (w  ++ " world!")

-- Function to check if a Python object has a specific attribute
checkPythonAttr :: Ptr () -> CString -> IO CInt
checkPythonAttr pyObj attrName = do
    -- Try to get the attribute
    attr <- pyGetAttrString pyObj attrName
    if attr == nullPtr
        then return 0  -- Attribute doesn't exist
        else do
            -- Check if the attribute is True/truthy
            result <- pyObjectIsTrue attr
            return result

foreign export ccall
    addFortyTwo :: CInt -> IO CInt

foreign export ccall
    addTenFloat :: CFloat -> IO CFloat

foreign export ccall
    addTenFloatWithAdd :: CFloat -> IO CFloat

foreign export ccall
    appendWorld :: CString -> IO CString

foreign export ccall
    checkPythonAttr :: Ptr () -> CString -> IO CInt