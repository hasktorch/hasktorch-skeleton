module HasktorchSkeleton
       ( someFunc
       ) where

import Torch.Typed.NN ()

someFunc :: IO ()
someFunc = putStrLn ("someFunc" :: String)
