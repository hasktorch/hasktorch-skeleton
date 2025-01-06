# Hasktorch Skeleton

Similar to the [Cardano Skeleton](https://github.com/input-output-hk/cardano-skeleton),
this repository serves as an example of how a downstream user of both Nix and Hasktorch
can set up a development environment.

The Nix shell installs ghc with hasktorch. When there is a hasktorch cache in nixpkgs, building hasktorch will be skipped.

# 3 Steps to happy Hasktorch coding

1. Fork this repo and clone it locally.
2. Launch a Nix shell with (optionally) CUDA, `hls`, and VS Code, `nix develop"`
3. Install the [Haskell Language Server plugin](https://marketplace.visualstudio.com/items?itemName=alanz.vscode-hie-server) and set the `HIE Variant` to `ghcide`.

Happy Hasktorch hacking!
