{
  nixConfig = {
    extra-substituters = [
      "https://hasktorch.cachix.org"
    ];
    extra-trusted-public-keys = [
      "hasktorch.cachix.org-1:wLjNS6HuFVpmzbmv01lxwjdCOtWRD8pQVR3Zr/wVoQc="
    ];
  };

  inputs = {
    hasktorch.url = "github:hasktorch/hasktorch";
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.follows = "hasktorch/nixpkgs";
  };

  outputs = inputs @ {
    self,
    hasktorch,
    nixpkgs,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      perSystem = {
        config,
        system,
        pkgs,
        ...
      }: let
        ghc = "ghc965";
      in {
        _module.args = {
          pkgs = import nixpkgs {
            inherit system;
            # Set to false to disable CUDA support
            config.cudaSupport = true;
            overlays = [
              hasktorch.overlays.default
            ];
          };
        };
        packages.default =
          pkgs.haskell.packages.${ghc}.callCabal2nix "example" ./. {};
        devShells.default = pkgs.haskell.packages.${ghc}.shellFor {
          packages = ps: [
            (ps.callCabal2nix "example" ./. {})
          ];
          nativeBuildInputs = with pkgs; [cabal-install stylish-haskell];
          withHoogle = true;
        };
      };
    };
}
