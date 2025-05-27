{-# LANGUAGE ForeignFunctionInterface #-}
module HaskellInterface (
    -- Re-export all functions that need to be called from Python
    addFortyTwo,
    addTenFloat,
    addTenFloatWithAdd,
    appendWorld,
    traverseTreeBFS,
    traverseJaxpr,
    checkPythonAttr,
    hsInitPython,
    hsCleanupPython
) where

import Foreign.C.Types
import Foreign.C.String
import Foreign.Ptr
import qualified HaskellTranspiler.Helpers as H
import qualified HaskellTranspiler.Math as M
import qualified HaskellTranspiler.Tree as T
import qualified HaskellTranspiler.Jaxpr as J

-- Re-export functions from Math module
addFortyTwo :: CInt -> IO CInt
addFortyTwo = M.addFortyTwo

addTenFloat :: CFloat -> IO CFloat
addTenFloat = M.addTenFloat

addTenFloatWithAdd :: CFloat -> IO CFloat
addTenFloatWithAdd = M.addTenFloatWithAdd

appendWorld :: CString -> IO CString
appendWorld = M.appendWorld

-- Re-export functions from Tree module
traverseTreeBFS :: Ptr () -> IO (Ptr ())
traverseTreeBFS = T.traverseTreeBFS

-- Re-export functions from Jaxpr module
traverseJaxpr :: Ptr () -> IO CInt
traverseJaxpr = J.traverseJaxpr

-- Check if a Python object has a specific attribute
checkPythonAttr :: Ptr () -> CString -> IO CInt
checkPythonAttr pyObj attrName = H.withGIL $ do
    H.withPyObject pyObj $ \obj -> do
        attr <- H.pyGetAttrString obj attrName
        if attr == nullPtr
            then do
                H.pyErrClear  -- Clear the exception if attribute doesn't exist
                return 0
            else do
                isTrue <- H.pyObjectIsTrue attr
                H.pyDecRef attr  -- Clean up the attribute
                return isTrue

-- Initialize Python when the module is loaded
foreign export ccall "hs_init_python" hsInitPython :: IO ()
hsInitPython = H.pyInitialize

-- Cleanup Python when the module is unloaded
foreign export ccall "hs_cleanup_python" hsCleanupPython :: IO ()
hsCleanupPython = H.pyFinalize

-- Export all functions that need to be called from Python
foreign export ccall
    addFortyTwo :: CInt -> IO CInt

foreign export ccall
    addTenFloat :: CFloat -> IO CFloat

foreign export ccall
    addTenFloatWithAdd :: CFloat -> IO CFloat

foreign export ccall
    appendWorld :: CString -> IO CString

foreign export ccall
    traverseTreeBFS :: Ptr () -> IO (Ptr ())

foreign export ccall
    traverseJaxpr :: Ptr () -> IO CInt

foreign export ccall
    checkPythonAttr :: Ptr () -> CString -> IO CInt 