############################################################################
# two-layer-network Nix build
############################################################################

{ system ? builtins.currentSystem
, crossSystem ? null
# allows to customize ghc and profiling (see ./nix/haskell.nix):
, config ? {}
# allows to override dependencies of the two-layer-network project without modifications
, sourcesOverride ? {}
# If true, activates CUDA support
, cudaSupport ? false
# If cudaSupport is true, this needs to be set to a valid CUDA major version number, e.g. 10:
# nix-build --arg cudaSupport true --argstr cudaMajorVersion 10
, cudaMajorVersion ? null
# pinned version of nixpkgs augmented with various overlays including hasktorch
, pkgs ? import ./nix/default.nix {
    inherit system crossSystem config sourcesOverride cudaSupport cudaMajorVersion;
  }
}:

# commonLib include iohk-nix utilities, our util.nix and nixpkgs lib.
with pkgs; with commonLib;

let

  haskellPackages = recRecurseIntoAttrs
    # the Haskell.nix package set, reduced to local packages.
    (selectProjectPackages twoLayerNetworkHaskellPackages);

  libs = collectComponents' "library" haskellPackages;
  exes = collectComponents' "exes" haskellPackages;

  self = {
    inherit haskellPackages;

    inherit (haskellPackages.two-layer-network.identifier) version;

    # Grab library components of this package.
    inherit (libs);

    # Grab executable components of this package.
    inherit (exes)
      two-layer-network
      ;

    # `tests` are the test suites which have been built.
    tests = collectComponents' "tests" haskellPackages;
    # `benchmarks` (only built, not run).
    benchmarks = collectComponents' "benchmarks" haskellPackages;

    checks = recurseIntoAttrs {
      # `checks.tests` collect results of executing the tests:
      tests = collectChecks haskellPackages;
    };

    shell = import ./shell.nix {
      inherit pkgs;
      withHoogle = true;
    };

    # Building the stack shell doesn't work in the sandbox. Pass `--option sandbox relaxed` or
    # `--option sandbox false` to be able to build this. You have to be root in order to that.
    # stackShell = import ./nix/stack-shell.nix {
    #   inherit pkgs;
    # };
  };
in
  self
