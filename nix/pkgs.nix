pkgs: _: with pkgs; {
  hasktorchSkeletonHaskellPackages = import ./haskell.nix {
    inherit
      lib
      stdenv
      pkgs
      haskell-nix
      buildPackages
      config
      ;
  };
}