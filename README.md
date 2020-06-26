# Hasktorch Skeleton

Similar to the [Cardano Skeleton](https://github.com/input-output-hk/cardano-skeleton),
this repository serves as an example of how a downstream user of both Nix and Hasktorch
can set up a development environment.

# 3 Steps to happy Hasktorch coding

1. Fork this repo and clone it locally.
2. Launch a Nix shell with (optionally) CUDA, `ghcide`, and VS Code, `nix-shell --arg cudaSupport true --argstr cudaMajorVersion 10 --command "code ."`
3. Install the [Haskell Language Server plugin](https://marketplace.visualstudio.com/items?itemName=alanz.vscode-hie-server) and set the `HIE Variant` to `ghcide`.

Happy Hasktorch hacking!
