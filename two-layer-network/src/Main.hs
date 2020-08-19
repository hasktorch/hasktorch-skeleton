module Main where

import Prelude hiding (atan, filter)
import qualified Control.Foldl as L
import Control.Lens (element, view, (^?))
import Control.Monad (ap, replicateM, void)
import Control.Monad.Trans.Class (MonadTrans (lift))
import Control.Monad.Trans.Cont (ContT (..), evalContT)
import Control.Monad.Trans.State (StateT (..), evalStateT, gets, state)
import Data.Foldable (foldrM)
import Data.Maybe (listToMaybe)
import Data.Random.Normal (normal)
import qualified Data.Set as Set
import Data.Text (Text, unpack)
import GHC.Exts (IsList (..), Item)
import GHC.Float (float2Double)
import GHC.Generics
import GHC.TypeLits
import Graphics.Vega.VegaLite hiding (name, sample)
import qualified Pipes as P
import qualified Pipes.Concurrent as P
import qualified Pipes.Group as P
import qualified Pipes.Prelude as P
import System.Random (Random, RandomGen, getStdGen, random)
import Torch.Data.Internal (fromInput', toOutput', withBufferLifted)
import Torch.Data.Pipeline (Dataset (..), MapStyleOptions (..), Sample (..), makeListT, mapStyleOpts)
import Torch.Data.StreamedPipeline (ListT (enumerate), MonadBaseControl)
import Torch.Typed hiding (DType, Device, shape, sin)

type DType = 'Float

type Device = '( 'CPU, 0)

-- type Device = '( 'CUDA, 0)

sinc :: Floating a => a -> a
sinc a = sin a / a

noisySinc :: (Floating a, Random a, RandomGen g) => a -> a -> g -> (a, g)
noisySinc eps a g = let (noise, g') = normal g in (sinc a + eps * noise, g')

data SincData = SincData {name :: Text, unSincData :: [(Float, Float)]} deriving (Eq, Ord)

mkSincData :: (RandomGen g, Monad m) => Text -> Int -> StateT g m SincData
mkSincData name size =
  let next = do
        x <- (* 20) <$> state normal
        y <- state (noisySinc 0.05 x)
        pure (x, y)
   in SincData name <$> replicateM size next

instance MonadFail m => Dataset m SincData Int (Float, Float) where
  getItem (SincData _ d) k = maybe (fail "invalid key") pure $ d ^? element k
  keys (SincData _ d) = Set.fromList [0 .. Prelude.length d -1]

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
    forward linear2 . atan . forward linear1

train ::
  _ =>
  model ->
  optim ->
  LearningRate device dtype ->
  ListT IO (Tensor device dtype shape, Tensor device dtype shape, Int) ->
  IO (model, optim)
train model optim learningRate examples =
  let step (model, optim) (x, y, _iter) = do
        let y' = forward model x
            loss = mseLoss @'ReduceMean y' y
        runStep model optim loss learningRate
      init = pure (model, optim)
      done = pure
   in P.foldM step init done . enumerate $ examples

evaluate ::
  forall m model device batchSize shape.
  _ =>
  model ->
  ListT m (Tensor device DType (batchSize ': shape), Tensor device DType (batchSize ': shape), Int) ->
  m Float
evaluate model examples =
  let step (loss, _) (x, y, iter) = do
        let y' = forward model x
            loss' = toFloat $ mseLoss @'ReduceMean y' y
        pure (loss + loss', iter)
      init = pure (0, 0)
      done (_, 0) = pure 0
      done (loss, iter) =
        let scale x = x / (fromInteger . toInteger $ iter + natValI @batchSize)
         in pure $ scale loss
   in P.foldM step init done . enumerate $ examples

infer ::
  _ =>
  model ->
  ListT m (Tensor device DType shape) ->
  m [([Float], [Float])]
infer model xs =
  let step rows x = do
        let y' = forward model $ x
        pure (rows ++ zip (toList . Just $ x) (toList . Just $ y'))
      init = pure []
      done = pure
   in P.foldM step init done . enumerate $ xs

type BatchSize = 100

type NumHidden = 100

main :: IO ()
main = do
  model <-
    TwoLayerNet @1 @1 @NumHidden
      <$> sample LinearSpec
      <*> sample LinearSpec

  let optim = mkAdam 0 0.9 0.999 (flattenParameters model)
      learningRate = 1e-2
      numEpochs = 100
      numWarmupEpochs = 10
      learningRateSchedule epoch
        | 0 < epoch && epoch <= numWarmupEpochs =
          let a :: Float = fromIntegral epoch / fromIntegral numWarmupEpochs
           in mulScalar a learningRate
        | numWarmupEpochs < epoch && epoch < numEpochs =
          let a :: Float = fromIntegral (numEpochs - epoch) / fromIntegral (numEpochs - numWarmupEpochs)
           in mulScalar a learningRate
        | otherwise = 0

  (trainingData, evaluationData, options) <-
    getStdGen
      >>= evalStateT
        ( (,,)
            <$> mkSincData "training" 10000
              <*> mkSincData "evaluation" 500
              <*> gets (\g -> (mapStyleOpts 1) {shuffle = Shuffle g})
        )

  let stats model trainingLosses evaluationLosses options = do
        (trainingLoss, options) <- evaluate' model options trainingData
        (evaluationLoss, options) <- evaluate' model options evaluationData
        trainingLosses <- pure $ trainingLoss : trainingLosses
        evaluationLosses <- pure $ evaluationLoss : evaluationLosses
        plot "plot.html" (infer' model) trainingLosses evaluationLosses evaluationData
        pure (trainingLosses, evaluationLosses, options)
      step (model, optim, trainingLosses, evaluationLosses, options) epoch = do
        (model, optim, options) <- train' model optim (learningRateSchedule epoch) options trainingData
        (trainingLosses, evaluationLosses, options) <- stats model trainingLosses evaluationLosses options
        pure (model, optim, trainingLosses, evaluationLosses, options)
      init = do
        (trainingLosses, evaluationLosses, options) <- stats model [] [] options
        pure (model, optim, trainingLosses, evaluationLosses, options)
      done = pure
  void . evalContT . P.foldM step init done . P.each $ [1 .. numEpochs]

train' :: _ => model -> optim -> LearningRate device dtype -> MapStyleOptions -> dataset -> ContT r IO (model, optim, MapStyleOptions)
train' model optim learningRate options sincData = do
  (examples, shuffle) <- makeListT options sincData
  (model, optim) <-
    lift . train model optim learningRate
      =<< collate @BatchSize 1 examples
  pure (model, optim, options {shuffle = shuffle})

evaluate' :: _ => model -> MapStyleOptions -> SincData -> ContT r IO (Float, MapStyleOptions)
evaluate' model options sincData = do
  (examples, shuffle) <- makeListT options sincData
  loss <-
    lift . evaluate model
      =<< collate @BatchSize @Device 1 examples
  lift . putStrLn $ "Average " <> unpack (name sincData) <> " loss: " <> show loss
  pure (loss, options {shuffle = shuffle})

infer' :: _ => model -> [Float] -> ContT r IO [([Float], [Float])]
infer' model xs =
  lift . infer model
    =<< collate' @BatchSize @Device @DType 1 (P.Select . P.each $ xs)

plot ::
  _ =>
  FilePath ->
  ([Float] -> ContT r IO [([Float], [Float])]) ->
  [Float] ->
  [Float] ->
  SincData ->
  ContT r IO ()
plot file infer trainingLosses evaluationLosses sincData = do
  let xs = [-20, -19.95 .. 20]
      mkDataRow x y =
        dataRow
          [ ("x", Number . float2Double $ x),
            ("y", Number . float2Double $ y)
          ]
  prediction <- do
    rawRows <- infer xs
    let f rawRow dataRows = do
          case rawRow of
            ([x], [y]) -> pure $ mkDataRow x y dataRows
            _ -> fail "invalid list shape(s)"
     in dataFromRows [] <$> foldrM f [] rawRows
  let evaluation = dataFromRows [] . foldr (uncurry mkDataRow) [] . unSincData $ sincData
  let target = dataFromRows [] . foldr (mkDataRow `ap` sinc) [] $ xs
  let losses =
        let mkDataRows title losses =
              zipWith
                ( \epoch loss ->
                    dataRow
                      [ ("epoch", Number . fromInteger $ epoch),
                        ("title", Str title),
                        ("loss", Number . float2Double $ loss)
                      ]
                )
                [0, 1 ..]
                (reverse losses)
         in dataFromRows [] . foldr id [] $
              mkDataRows "training" trainingLosses
                <> mkDataRows "evaluation" evaluationLosses
  lift . toHtmlFile file . mkVegaLite . datasets $
    [ (("prediction", prediction)),
      (("evaluation", evaluation)),
      (("target", target)),
      (("losses"), losses)
    ]

collate ::
  forall batchSize device dtype r m.
  (KnownNat batchSize, KnownDevice device, ComputeHaskellType dtype ~ Float, MonadBaseControl IO m) =>
  Int ->
  ListT m ((Float, Float), Int) ->
  ContT r m (ListT m (Tensor device dtype '[batchSize, 1], Tensor device dtype '[batchSize, 1], Int))
collate n = bufferedCollate (P.bounded n) (natValI @batchSize) f
  where
    f exs =
      let (xys, iters) = unzip exs
          (xs, ys) = unzip xys
       in (,,) <$> fromList (pure <$> xs) <*> fromList (pure <$> ys) <*> listToMaybe iters

collate' ::
  forall batchSize device dtype r m.
  (KnownNat batchSize, KnownDevice device, ComputeHaskellType dtype ~ Float, MonadBaseControl IO m) =>
  Int ->
  ListT m Float ->
  ContT r m (ListT m (Tensor device dtype '[batchSize, 1]))
collate' n = bufferedCollate (P.bounded n) (natValI @batchSize) f
  where
    f xs = fromList (pure <$> xs)

bufferedCollate ::
  forall a b r m.
  MonadBaseControl IO m =>
  P.Buffer b ->
  Int ->
  ([a] -> Maybe b) ->
  ListT m a ->
  ContT r m (ListT m b)
bufferedCollate buffer batchSize f as = ContT $ \g ->
  snd
    <$> withBufferLifted
      buffer
      fOutput
      (g . P.Select . fromInput')
  where
    fOutput output = P.runEffect $ (P.>-> (toOutput' output)) . (P.>-> P.mapMaybe f) . L.purely P.folds L.list . view (P.chunksOf batchSize) . enumerate $ as

mkVegaLite :: Data -> VegaLite
mkVegaLite dataset =
  let -- width and height of the individual plots (in pixels)
      w = width 600
      h = height 400

      encOverview =
        encoding
          . position X [PName "x", PmType Quantitative]
          . position Y [PName "y", PmType Quantitative, PScale scaleOptsOverview]
      scaleOptsOverview = [SDomain (DNumbers [-2, 2]), SNice (IsNice False)]
      transOverview =
        transform
          . filter (FRange "x" (NumberRange (-20) 20))
          . filter (FRange "y" (NumberRange (-2) 2))
      target =
        asSpec
          [ dataFromSource "target" [],
            transOverview [],
            mark Line [MStrokeWidth 0.5, MStroke "black"]
          ]
      evaluation =
        asSpec
          [ dataFromSource "evaluation" [],
            transOverview [],
            mark Point [MSize 5, MStroke "black"]
          ]
      prediction =
        asSpec
          [ dataFromSource "prediction" [],
            transOverview [],
            mark Line []
          ]
      overview =
        asSpec
          [ layer [target, evaluation, prediction],
            encOverview [],
            w,
            h
          ]

      encLosses =
        encoding
          . position X [PName "epoch", PmType Quantitative, PScale scaleOptsLosses]
          . position Y [PName "loss", PmType Quantitative, PScale [SType ScLog]]
          . color [MName "title", MmType Nominal, MLegend [LTitle "", LOrient LOBottom]]
      scaleOptsLosses = [SDomain (DNumbers [0, 100]), SNice (IsNice False)]
      losses =
        asSpec
          [ dataFromSource "losses" [],
            encLosses [],
            mark Line [],
            w,
            h
          ]
   in toVegaLite
        [ dataset,
          vConcat [overview, losses]
        ]
