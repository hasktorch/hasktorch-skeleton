{
  nixConfig = {
    bash-prompt = "\[hasktorch-skeleton$(__git_ps1 \" (%s)\")\]$ ";
  };
  inputs = {
    utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs?rev=01d7c7caba0f021e986f7e46fae3c8e41265a145";
  };

  outputs = { self, nixpkgs, utils  }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = system == "x86_64-linux";
          # config.ihaskell.packages = pkgs: with pkgs; [
          #   hasktorch
          # ];
        };
        ghcWithHasktorch = pkgs.haskellPackages.ghcWithPackages (pkgs: with pkgs; [
          hasktorch
          haskell-language-server
        ]);
      in {
        defaultPackage = pkgs.haskellPackages.callCabal2nix "hasktorch-skeleton" ./. {};
        devShell = with pkgs; mkShell {
          buildInputs = [
            ghcWithHasktorch
            cabal-install
            stack
            # ihaskell
          ];
          shellHook = ''
            source ${git}/share/bash-completion/completions/git-prompt.sh
          '';
        };
      });
}
