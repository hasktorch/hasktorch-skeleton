module Main where

import qualified Control.Foldl as L
import Control.Lens (element, view, (^?))
import Control.Monad (foldM_, replicateM, void, when)
import Control.Monad.Trans.Class (MonadTrans (lift))
import Control.Monad.Trans.Cont (ContT (..), cont, evalContT)
import Control.Monad.Trans.State (StateT, evalStateT, gets, runState, runStateT, state)
import Data.Maybe (listToMaybe)
import Data.Random.Normal (normal)
import qualified Data.Set as Set
import GHC.Exts (IsList (fromList))
import GHC.Generics
import GHC.TypeLits
import qualified Pipes as P
import qualified Pipes.Concurrent as P
import qualified Pipes.Group as P
import qualified Pipes.Prelude as P
import System.Random (Random, RandomGen, getStdGen, random)
import Torch.Data.Internal (fromInput', toOutput', withBufferLifted)
import Torch.Data.Pipeline (Dataset (..), MapStyleOptions (..), Sample (..), makeListT, mapStyleOpts)
import Torch.Data.StreamedPipeline (ListT (enumerate), MonadBaseControl)
import Torch.Internal.Managed.Type.Context (manual_seed_L)
import Torch.Typed hiding (DType, Device, sin)

type DType = 'Float

type Device = '( 'CPU, 0)

type BatchSize = 128

type NumHidden = 100

noisySinc :: (Random a, Floating a, RandomGen g) => a -> g -> (a, g)
noisySinc a g = let (noise, g') = random g in (sin a / a + noise, g')

data SincData = SincData [(Float, Float)] deriving (Eq, Ord)

mkSincData :: (RandomGen g, Monad m) => Int -> StateT g m SincData
mkSincData size =
  let next = do
        x <- (* 10) <$> state normal
        y <- state $ noisySinc x
        pure (x, y)
   in SincData <$> replicateM size next

instance MonadFail m => Dataset m SincData Int (Float, Float) where
  getItem (SincData d) k = maybe (fail "invalid key") pure $ d ^? element k
  keys (SincData d) = Set.fromList [0 .. Prelude.length d -1]

data TwoLayerNet (numIn :: Nat) (numOut :: Nat) (numHidden :: Nat) = TwoLayerNet
  { linear1 :: Linear numIn numHidden DType Device,
    linear2 :: Linear numHidden numOut DType Device
  }
  deriving (Show, Generic)

instance
  HasForward
    (TwoLayerNet numIn numOut numHidden)
    (Tensor Device DType '[batchSize, numIn])
    (Tensor Device DType '[batchSize, numOut])
  where
  forward TwoLayerNet {..} =
    forward linear2 . relu . forward linear1

main :: IO ()
main = do
  model <-
    TwoLayerNet @1 @1 @NumHidden
      <$> sample LinearSpec
      <*> sample LinearSpec

  let optim = mkAdam 0 0.9 0.999 (flattenParameters model)

  (trainingData, evaluationData, options) <-
    getStdGen
      >>= evalStateT
        ( (,,)
            <$> mkSincData 4096
              <*> mkSincData 256
              <*> gets (\g -> (mapStyleOpts 1) {shuffle = Shuffle g})
        )

  let learningRate = 1e-4
      train model optim examples =
        let step (model, optim) (x, y, iter) = do
              -- putStrLn $ "Training batch " <> show iter
              let y' = forward model x
                  loss = mseLoss @'ReduceMean y' y
              runStep model optim loss learningRate
            init = pure (model, optim)
            done = pure
         in P.foldM step init done . enumerate $ examples

  let evaluate model examples =
        let step (loss, _) (x, y, iter) = do
              let y' = forward model x
                  loss' = toFloat $ mseLoss @'ReduceMean y' y
              pure (loss + loss', iter)
            init = pure (0, 0)
            done (_, 0) = pure 0
            done (loss, iter) =
              let scale x = x / (fromInteger . toInteger $ iter + natValI @BatchSize)
               in pure $ scale loss
         in P.foldM step init done . enumerate $ examples

  let numEpochs = 100 :: Int
      eval model options = do
        (examples, shuffle) <- makeListT options evaluationData
        loss <-
          lift . evaluate model
            =<< collate @BatchSize 1 examples
        lift . putStrLn $ "Average evaluation loss: " <> show loss
        pure shuffle
      step (model, optim, options) epoch = do
        (examples, shuffle) <- makeListT options trainingData
        (model', optim') <-
          lift . train model optim
            =<< collate @BatchSize 1 examples
        shuffle <- eval model options {shuffle = shuffle}
        pure (model', optim', options {shuffle = shuffle})
      init = do
        shuffle <- eval model options
        pure (model, optim, options {shuffle = shuffle})
      done = pure

  void . evalContT $ P.foldM step init done (P.each [1 .. numEpochs])

collate ::
  forall batchSize device dtype r m.
  (KnownNat batchSize, KnownDevice device, ComputeHaskellType dtype ~ Float, Monad m, MonadBaseControl IO m) =>
  Int ->
  ListT m ((Float, Float), Int) ->
  ContT r m (ListT m (Tensor Device DType '[batchSize, 1], Tensor device dtype '[batchSize, 1], Int))
collate n examples = ContT $ \g ->
  snd
    <$> withBufferLifted
      (P.bounded n)
      fOutput
      (g . P.Select . fromInput')
  where
    f exs =
      let (xys, iters) = unzip exs
          (xs, ys) = unzip xys
       in (,,) <$> fromList (pure <$> xs) <*> fromList (pure <$> ys) <*> listToMaybe iters
    fOutput :: P.Output (Tensor Device DType '[batchSize, 1], Tensor device dtype '[batchSize, 1], Int) -> m ()
    fOutput output = P.runEffect $ (P.>-> (toOutput' output)) . (P.>-> P.mapMaybe f) . L.purely P.folds L.list . view (P.chunksOf (natValI @BatchSize)) . enumerate $ examples
