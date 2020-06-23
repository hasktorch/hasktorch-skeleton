{ lib
, stdenv
, pkgs
, haskell-nix
, buildPackages
, config ? {}
# GHC attribute name
, compiler ? config.haskellNix.compiler or "ghc883"
# Enable profiling
, profiling ? config.haskellNix.profiling or false
# Enable CUDA support
, cudaSupport ? false
}:

let

  src = haskell-nix.haskellLib.cleanGit {
      name = "hasktorch-skeleton";
      src = ../.;
  };

  # This creates the Haskell package set.
  # https://input-output-hk.github.io/haskell.nix/user-guide/projects/
  pkgSet = haskell-nix.cabalProject {
    inherit src;

    compiler-nix-name = compiler;

    # these extras will provide additional packages
    # ontop of the package set derived from cabal resolution.
    pkg-def-extras = [ ];

    modules = [
      # Fixes for libtorch-ffi
      {
        packages.libtorch-ffi = {
          configureFlags = [
            "--extra-lib-dirs=${pkgs.torch}/lib"
            "--extra-include-dirs=${pkgs.torch}/include"
            "--extra-include-dirs=${pkgs.torch}/include/torch/csrc/api/include"
          ];
          flags = {
            cuda = cudaSupport;
            gcc = !cudaSupport && pkgs.stdenv.hostPlatform.isDarwin;
          };
        };
      }
    ];
  };

in
  pkgSet