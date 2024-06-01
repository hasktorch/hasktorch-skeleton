# Hasktorch Skeleton

Similar to the [Cardano Skeleton](https://github.com/input-output-hk/cardano-skeleton),
this repository serves as an example of how a downstream user of both Nix and Hasktorch
can set up a development environment.

# 3 Steps to happy Hasktorch coding

1. Fork this repo and clone it locally.
2. (optional) Disable CUDA by editing `flake.nix` and setting
   `cudaSupport` to `false`.
3. Start a Nix development shell with `nix develop .`.

Happy Hasktorch hacking!
