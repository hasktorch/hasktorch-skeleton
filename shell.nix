# This file is used by nix-shell.
# It just takes the shell attribute from default.nix.
{ config ? {}
, sourcesOverride ? {}
# If true, activates CUDA support
, cudaSupport ? false
# If cudaSupport is true, this needs to be set to a valid CUDA major version number, e.g. 10:
# nix-shell --arg cudaSupport true --argstr cudaMajorVersion 10
, cudaMajorVersion ? null
, withHoogle ? false
, pkgs ? import ./nix/default.nix {
    inherit config sourcesOverride cudaSupport cudaMajorVersion;
  }
}:
with pkgs;
let
  # This provides a development environment that can be used with nix-shell or
  # lorri. See https://input-output-hk.github.io/haskell.nix/user-guide/development/
  shell = twoLayerNetworkHaskellPackages.shellFor {
    name = "two-layer-network-dev-shell";

    tools = {
      cabal = "3.2.0.0";
      haskell-language-server = "0.6.0";
    };

    # Prevents cabal from choosing alternate plans, so that
    # *all* dependencies are provided by Nix.
    # TODO: Set to true as soon as haskell.nix issue #231 is resolved.
    exactDeps = false;

    shellHook =
      let
        cpath = ''
          export CPATH=${torch}/include/torch/csrc/api/include
        '';
        nproc = ''
          case "$(uname)" in
            "Linux")
                ${pkgs.utillinux}/bin/taskset -pc 0-1000 $$
            ;;
          esac
        '';
        libraryPath = stdenv.lib.optionalString cudaSupport ''
          export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib"
        '';
      in
        cpath + nproc + libraryPath;

    inherit withHoogle;
  };

in

 shell
