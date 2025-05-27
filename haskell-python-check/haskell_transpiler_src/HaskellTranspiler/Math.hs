{-# LANGUAGE ForeignFunctionInterface #-}
module HaskellTranspiler.Math where

import Foreign.C.Types
import Foreign.C.String

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
    newCString (w ++ " world!") 