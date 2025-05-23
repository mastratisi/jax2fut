{-# LANGUAGE ForeignFunctionInterface #-}
module Test where
 
import Foreign.C.Types
import Foreign.C.String
import Foreign.Ptr
import Foreign.StablePtr
import Foreign.Marshal.Alloc
import Control.Exception (bracket)

-- Import Python C API functions
foreign import ccall "Python.h Py_Initialize" pyInitialize :: IO ()
foreign import ccall "Python.h Py_Finalize" pyFinalize :: IO ()
foreign import ccall "Python.h PyGILState_Ensure" pyGILStateEnsure :: IO CInt
foreign import ccall "Python.h PyGILState_Release" pyGILStateRelease :: CInt -> IO ()
foreign import ccall "Python.h PyObject_GetAttrString" pyGetAttrString :: Ptr () -> CString -> IO (Ptr ())
foreign import ccall "Python.h PyObject_IsTrue" pyObjectIsTrue :: Ptr () -> IO CInt
foreign import ccall "Python.h Py_DecRef" pyDecRef :: Ptr () -> IO ()
foreign import ccall "Python.h Py_IncRef" pyIncRef :: Ptr () -> IO ()
foreign import ccall "Python.h PyErr_Clear" pyErrClear :: IO ()

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

-- Safe wrapper for GIL management
withGIL :: IO a -> IO a
withGIL action = do
    gilState <- pyGILStateEnsure
    result <- action
    pyGILStateRelease gilState
    return result

-- Safe wrapper for Python object reference management
withPyObject :: Ptr () -> (Ptr () -> IO a) -> IO a
withPyObject obj action = do
    pyIncRef obj
    result <- action obj
    pyDecRef obj
    return result

-- Function to check if a Python object has a specific attribute
checkPythonAttr :: Ptr () -> CString -> IO CInt
checkPythonAttr pyObj attrName = withGIL $ do
    withPyObject pyObj $ \obj -> do
        -- Try to get the attribute
        attr <- pyGetAttrString obj attrName
        if attr == nullPtr
            then do
                pyErrClear  -- Clear the exception if attribute doesn't exist
                return 0  -- Attribute doesn't exist
            else do
                -- Check if the attribute is True/truthy
                isTrue <- pyObjectIsTrue attr
                pyDecRef attr  -- Clean up the attribute
                return isTrue

-- Initialize Python when the module is loaded
foreign export ccall "hs_init_python" hsInitPython :: IO ()
hsInitPython = pyInitialize

-- Cleanup Python when the module is unloaded
foreign export ccall "hs_cleanup_python" hsCleanupPython :: IO ()
hsCleanupPython = pyFinalize

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