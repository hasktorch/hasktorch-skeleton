# Hasktorch Skeleton

Similar to the [Cardano Skeleton](https://github.com/input-output-hk/cardano-skeleton), this repository serves as an example of how a downstream user of both Nix and Hasktorch can set up a development environment.

The Nix shell installs GHC with Hasktorch. When there is a Hasktorch cache in nixpkgs, building Hasktorch will be skipped. Additionally, you can speed up builds by using Cachix.

## Steps to Happy Hasktorch Coding

1. Fork this repo and clone it locally.
2. Enable the Hasktorch binary cache using Cachix:
   ```sh
   cachix use hasktorch
   ```
   This works on both Linux and macOS, reducing build times significantly.
3. Launch a Nix shell, which includes GHC with Hasktorch and Haskell Language Server (hls):
   ```sh
   nix develop
   ```
4. Install the [Haskell Language Server plugin](https://marketplace.visualstudio.com/items?itemName=alanz.vscode-hie-server) and [direnv](https://github.com/direnv/direnv-vscode).

## Using IHaskell Notebook

To use `ihaskell-notebook`, uncomment `ihaskell` in the `flake.nix and restart the development shell:
```sh
nix develop
```
Then, launch Jupyter Notebook:
```sh
ihaskell-notebook
```
This allows you to use Haskell interactively in a Jupyter environment.

Happy Hasktorch hacking!
