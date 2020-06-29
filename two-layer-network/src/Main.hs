module Main where

import GHC.TypeLits (KnownNat, Nat)
import Torch.Typed.NN (LinearSpec (..), Linear)
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.NN as A
import GHC.Generics (Generic)
import Torch.Typed.Tensor (Tensor)
import Torch.Typed.Functional (mseLoss, relu)
import Torch.Typed.Factories (rand)
import Torch.Typed.Optim (mkGD, runStep)
import Torch.Typed.Parameter (Parameterized(flattenParameters))
import Control.Monad (when, foldM_)
import qualified Torch.Functional as D
import qualified Torch.Internal.Managed.Type.Context as ATen

type DType = 'D.Float
type Device = '( 'D.CPU, 0)

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
  A.HasForward
    (TwoLayerNet dIn dOut h)
    (Tensor Device DType '[bs, dIn])
    (Tensor Device DType '[bs, dOut])
  where
    forward TwoLayerNet {..} =
      A.forward linear2 . relu . A.forward linear1

train :: IO ()
train = do
  
  ATen.manual_seed_L 1
  x <- rand @'[N, DIn] @DType @Device
  y <- rand @'[N, DOut] @DType @Device

  model <-
    TwoLayerNet @DIn @DOut @H
      <$> A.sample LinearSpec
      <*> A.sample LinearSpec

  let learningRate = 1e-4
      optim = mkGD

  let step (model, optim) epoch = do
        let y' = A.forward model x
            loss = mseLoss @'D.ReduceMean y' y
        when (epoch `mod` 100 == 99) $
          putStrLn $ "epoch = " <> show epoch <> " " <> "loss = " <> show loss
        runStep model optim loss learningRate
      numEpoch = 500
  foldM_ step (model, optim) [1 .. numEpoch]

main :: IO ()
main = train
