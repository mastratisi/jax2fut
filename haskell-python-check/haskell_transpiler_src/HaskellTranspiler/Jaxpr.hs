{-# LANGUAGE ForeignFunctionInterface #-}
module HaskellTranspiler.Jaxpr where

import Foreign.C.Types
import Foreign.C.String
import Foreign.Ptr
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Storable
import qualified HaskellTranspiler.Helpers as H

-- Structure to represent a JAXPR node
data JaxprNode = JaxprNode
    { nodePrimitive :: String
    , nodeType :: String
    } deriving (Show)

-- Traverse a JAXPR and count nodes
traverseJaxpr :: Ptr () -> IO CInt
traverseJaxpr jaxprPtr = H.withGIL $ do
    -- Get the list of equations from the JAXPR
    eqnsAttr <- H.getPyAttr jaxprPtr "eqns"
    if eqnsAttr == nullPtr
        then do
            H.pyErrClear
            return 0
        else do
            -- Get the length of the equations list
            len <- H.pyListSize eqnsAttr
            H.pyDecRef eqnsAttr
            
            -- Process each equation
            processEquations jaxprPtr (fromIntegral len) 0
  where
    processEquations :: Ptr () -> CInt -> CInt -> IO CInt
    processEquations _ 0 count = return count
    processEquations jaxprPtr remaining count = do
        -- Get the equations list again for this iteration
        eqnsAttr <- H.getPyAttr jaxprPtr "eqns"
        if eqnsAttr == nullPtr
            then do
                H.pyErrClear
                return count
            else do
                -- Get the current equation using index
                let idx = fromIntegral (remaining - 1)
                eqn <- H.pyListGetItem eqnsAttr idx
                if eqn == nullPtr
                    then do
                        H.pyErrClear
                        H.pyDecRef eqnsAttr
                        return count
                    else do
                        -- Get the primitive
                        primitiveAttr <- H.getPyAttr eqn "primitive"
                        name <- if primitiveAttr == nullPtr
                            then do
                                H.pyErrClear
                                H.pyDecRef eqn
                                H.pyDecRef eqnsAttr
                                return "unknown"
                            else do
                                -- Get the primitive name
                                nameAttr <- H.getPyAttr primitiveAttr "name"
                                if nameAttr == nullPtr
                                    then do
                                        H.pyErrClear
                                        H.pyDecRef primitiveAttr
                                        H.pyDecRef eqn
                                        H.pyDecRef eqnsAttr
                                        return "unknown"
                                    else do
                                        cstr <- H.pyUnicodeAsUTF8 nameAttr
                                        name <- peekCString cstr
                                        H.pyDecRef nameAttr
                                        H.pyDecRef primitiveAttr
                                        H.pyDecRef eqn
                                        H.pyDecRef eqnsAttr
                                        return name
                        
                        -- Print node information
                        putStrLn $ "Found node: " ++ name
                        
                        -- Process next equation
                        processEquations jaxprPtr (remaining - 1) (count + 1)

-- Foreign export for Python
foreign export ccall "traverse_jaxpr" traverseJaxpr :: Ptr () -> IO CInt 