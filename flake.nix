{
  inputs = {
    haskell-nix.url = "github:input-output-hk/haskell.nix";
    stackageSrc = {
      url = "github:input-output-hk/stackage.nix";
      flake = false;
    };
    hackageSrc = {
      url = "github:input-output-hk/hackage.nix";
      flake = false;
    };
    utils.url = "github:numtide/flake-utils";
    libtorch-nix = {
      url = "github:hasktorch/libtorch-nix/d14b0fd10b96d192b92b8ccc7254ade4b3489331";
      flake = false;
    };
  };

  outputs = inputs@{ self, nixpkgs, haskell-nix, utils, ... }:
    let
      name = "hasktorchSkeleton";
      compiler = "ghc8104"; # Not used for `stack.yaml` based projects.
      cudaSupport = false;
      cudaMajorVersion = null;
      project-name = "${name}HaskellPackages";


      # This overlay adds our project to pkgs
      project-overlay = final: prev: {
        ${project-name} =
            #assert compiler == supported-compilers;
            final.haskell-nix.project' {
              # 'cleanGit' cleans a source directory based on the files known by git
              src = prev.haskell-nix.haskellLib.cleanGit {
                inherit name;
                src = ./.;
              };

              compiler-nix-name = compiler;
              projectFileName = "cabal.project"; # Not used for `stack.yaml` based projects.
              modules = [
                # Fixes for libtorch-ffi
                {
                  packages.libtorch-ffi = {
                    configureFlags = with final; [
                      "--extra-lib-dirs=${torch}/lib"
                      "--extra-include-dirs=${torch}/include"
                      "--extra-include-dirs=${torch}/include/torch/csrc/api/include"
                    ];
                    flags = {
                      cuda = cudaSupport;
                      gcc = !cudaSupport && final.stdenv.hostPlatform.isDarwin;
                    };
                  };
                }
              ];

            };

      };
    in
      { overlay = final: prev: {
          "${name}" = ("${project-name}-overlay" final prev)."${project-name}".flake {};
        };
      } // (utils.lib.eachSystem [ "x86_64-linux" ] (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [
              haskell-nix.overlay
              (final: prev: {
                haskell-nix = prev.haskell-nix // {
                  sources = prev.haskell-nix.sources // {
                    hackage = inputs.hackageSrc;
                    stackage = inputs.stackageSrc;
                  };
                  modules = [
                    # Fixes for libtorch-ffi
                    {
                      packages.libtorch-ffi = {
                        configureFlags = with final; [
                          "--extra-lib-dirs=${torch}/lib"
                          "--extra-include-dirs=${torch}/include"
                          "--extra-include-dirs=${torch}/include/torch/csrc/api/include"
                        ];
                        flags = {
                          cuda = cudaSupport;
                          gcc = !cudaSupport && final.stdenv.hostPlatform.isDarwin;
                        };
                      };
                    }
                  ];
                };
              })
              (import ./nix/overlays/libtorch.nix { inherit inputs cudaSupport cudaMajorVersion; })
              project-overlay
            ];
          };
          flake = pkgs."${project-name}".flake {};
        in flake // rec {

          packages.example = flake.packages."${name}:exe:example";

          defaultPackage = packages.example;

          devShell = (import ./shell.nix {
              inherit cudaSupport cudaMajorVersion pkgs;
              withHoogle = false;
            });
        }
        ));
}
