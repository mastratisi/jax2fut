{-# LANGUAGE ForeignFunctionInterface #-}
module HaskellTranspiler.Helpers where

import Foreign.C.Types
import Foreign.C.String
import Foreign.Ptr
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
foreign import ccall "Python.h PyList_New" pyListNew :: CInt -> IO (Ptr ())
foreign import ccall "Python.h PyList_Append" pyListAppend :: Ptr () -> Ptr () -> IO CInt
foreign import ccall "Python.h PyLong_FromLong" pyLongFromLong :: CLong -> IO (Ptr ())
foreign import ccall "Python.h PyObject_GetIter" pyObjectGetIter :: Ptr () -> IO (Ptr ())
foreign import ccall "Python.h PyIter_Next" pyIterNext :: Ptr () -> IO (Ptr ())
foreign import ccall "Python.h PyObject_GetAttr" pyObjectGetAttr :: Ptr () -> Ptr () -> IO (Ptr ())
foreign import ccall "Python.h PyLong_AsLong" pyLongAsLong :: Ptr () -> IO CLong

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

-- Helper function to get a Python object's attribute
getPyAttr :: Ptr () -> String -> IO (Ptr ())
getPyAttr obj attrName = do
    attrNamePtr <- newCString attrName
    attr <- pyGetAttrString obj attrNamePtr
    free attrNamePtr
    return attr

-- Helper function to get a Python object's value as an integer
getPyInt :: Ptr () -> IO (Maybe CInt)
getPyInt obj = do
    valueAttr <- getPyAttr obj "value"
    if valueAttr == nullPtr
        then do
            pyErrClear
            return Nothing
        else do
            value <- pyLongAsLong valueAttr
            pyDecRef valueAttr
            return $ Just (fromIntegral value)

-- Helper function to get a Python object's children as a list
getPyChildren :: Ptr () -> IO [Ptr ()]
getPyChildren obj = do
    childrenAttr <- getPyAttr obj "children"
    if childrenAttr == nullPtr
        then do
            pyErrClear
            return []
        else do
            iter <- pyObjectGetIter childrenAttr
            pyDecRef childrenAttr
            if iter == nullPtr
                then do
                    pyErrClear
                    return []
                else do
                    children <- getIterItems iter
                    pyDecRef iter
                    return children

-- Helper function to get all items from a Python iterator
getIterItems :: Ptr () -> IO [Ptr ()]
getIterItems iter = do
    item <- pyIterNext iter
    if item == nullPtr
        then do
            pyErrClear
            return []
        else do
            rest <- getIterItems iter
            return (item : rest) 