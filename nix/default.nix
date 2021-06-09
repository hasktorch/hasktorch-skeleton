{ system ? builtins.currentSystem
, crossSystem ? null
# Lets you customise ghc and profiling (see ./haskell.nix):
, config ? {}
# Lets you override niv dependencies of the project without modifications to the source.
, sourcesOverride ? {}
# Version info, to be passed when not building from a git work tree
, gitrev ? null
# Enable CUDA support
, cudaSupport ? false
, cudaMajorVersion ? null
# Add packages on top of the package set derived from cabal resolution
, extras ? (_: {})
}:

# assert that the correct cuda versions are used
assert cudaSupport -> (cudaMajorVersion == "9" || cudaMajorVersion == "10" || cudaMajorVersion == "11");

let
  sources = import ./sources.nix { inherit pkgs; }
    // sourcesOverride;
  iohkNix = import sources.iohk-nix {};
  haskellNix = import sources.haskell-nix { inherit system sourcesOverride; };

  # Use haskell.nix default nixpkgs
  nixpkgsSrc = haskellNix.sources.nixpkgs-unstable;

  # for inclusion in pkgs:
  overlays =
    # Haskell.nix (https://github.com/input-output-hk/haskell.nix)
    haskellNix.nixpkgsArgs.overlays
    # override Haskell.nix hackage and stackage sources
    ++ [
      (pkgsNew: pkgsOld: let inherit (pkgsNew) lib; in {
        haskell-nix = pkgsOld.haskell-nix // {
          hackageSrc = sources.hackage-nix;
          stackageSrc = sources.stackage-nix;
        };
      })
    ]
    # the haskell-nix.haskellLib.extra overlay contains some useful extra utility functions for haskell.nix
    ++ iohkNix.overlays.haskell-nix-extra
    # the iohkNix overlay contains nix utilities and niv
    ++ iohkNix.overlays.iohkNix
    ++ [
      (import ./overlays/libtorch.nix {
        inherit cudaSupport cudaMajorVersion;
        inputs = sources;
      })
    ]
    # our own overlays:
    ++ [
      (pkgs: _: with pkgs; {
        inherit gitrev cudaSupport extras;

        # commonLib: mix pkgs.lib with iohk-nix utils and sources:
        commonLib = lib // iohkNix
          // import ./util.nix { inherit haskell-nix; }
          # also expose sources, nixpkgs and overlays
          // { inherit overlays sources nixpkgsSrc; };
      })
      # haskell-nix-ified hasktorch cabal project:
      (import ./pkgs.nix)
    ];

  pkgs = import nixpkgsSrc {
    inherit system crossSystem overlays;
    config = haskellNix.nixpkgsArgs.config // config;
  };

in pkgs
