module Main where

import Control.Monad (foldM_, when)
import GHC.Generics
import GHC.TypeLits
import Torch.Internal.Managed.Type.Context (manual_seed_L)
import Torch.Typed hiding (DType, Device)

type DType = 'Float
type Device = '( 'CPU, 0)

type N = 64
type DIn = 1000
type H = 100
type DOut = 10

data TwoLayerNet (dIn :: Nat) (dOut :: Nat) (h :: Nat)
  = TwoLayerNet { linear1 :: Linear dIn h DType Device
                , linear2 :: Linear h dOut DType Device
                }
  deriving (Show, Generic)

instance
  HasForward
    (TwoLayerNet dIn dOut h)
    (Tensor Device DType '[bs, dIn])
    (Tensor Device DType '[bs, dOut])
  where
    forward TwoLayerNet {..} =
      forward linear2 . relu . forward linear1

main :: IO ()
main = do
  
  manual_seed_L 1
  x <- rand @'[N, DIn] @DType @Device
  y <- rand @'[N, DOut] @DType @Device

  model <-
    TwoLayerNet @DIn @DOut @H
      <$> sample LinearSpec
      <*> sample LinearSpec

  let learningRate = 1e-4
      optim = mkGD

  let step (model, optim) epoch = do
        let y' = forward model x
            loss = mseLoss @'ReduceMean y' y
        when (epoch `mod` 100 == 99) $
          putStrLn $ "epoch = " <> show epoch <> " " <> "loss = " <> show loss
        runStep model optim loss learningRate
      numEpoch = 500
  foldM_ step (model, optim) [1 .. numEpoch]
