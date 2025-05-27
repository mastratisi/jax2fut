{-# LANGUAGE ForeignFunctionInterface #-}
module HaskellTranspiler.Tree where

import Foreign.C.Types
import Foreign.Ptr
import HaskellTranspiler.Helpers

-- Breadth-first traversal of a Python tree
traverseTreeBFS :: Ptr () -> IO (Ptr ())
traverseTreeBFS root = withGIL $ do
    -- Create a Python list to store the result
    resultList <- pyListNew 0
    
    -- Initialize the queue with the root node
    let queue = [root]
    
    -- Process the queue
    processQueue queue resultList
    
    return resultList
  where
    processQueue :: [Ptr ()] -> Ptr () -> IO ()
    processQueue [] _ = return ()
    processQueue (node:rest) resultList = do
        -- Get the node's value and add it to the result list
        maybeValue <- getPyInt node
        case maybeValue of
            Just value -> do
                pyValue <- pyLongFromLong (fromIntegral value)
                pyListAppend resultList pyValue
                pyDecRef pyValue
            Nothing -> return ()
        
        -- Get the node's children and add them to the queue
        children <- getPyChildren node
        let newQueue = rest ++ children
        
        -- Process the rest of the queue
        processQueue newQueue resultList
        
        -- Clean up the current node
        pyDecRef node 