pkgs: _: with pkgs; {
  twoLayerNetworkHaskellPackages = import ./haskell.nix {
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
