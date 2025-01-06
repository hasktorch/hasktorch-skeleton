{
  nixConfig = {
    bash-prompt = "\[hasktorch-skeleton$(__git_ps1 \" (%s)\")\]$ ";
  };
  inputs = {
    utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:junjihashimoto/nixpkgs?rev=5c77923028d8ec6c54dc9820e44bced372c68f3d";
  };

  outputs = { self, nixpkgs, utils  }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = system == "x86_64-linux";
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
          ];
          shellHook = ''
            source ${git}/share/bash-completion/completions/git-prompt.sh
          '';
        };
      });
}
