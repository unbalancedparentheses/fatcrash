{
  description = "fatcrash — crash detection via fat-tail statistics";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          rust = pkgs.rust-bin.stable.latest.default.override {
            extensions = [ "rust-src" "rust-analyzer" ];
          };
          python = pkgs.python313;
        in
        {
          default = pkgs.mkShell {
            packages = [
              rust
              python
              pkgs.maturin
              pkgs.uv
            ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
              pkgs.libiconv
            ];

            env = {
              RUST_BACKTRACE = "1";
            };

            shellHook = ''
              if [ ! -d .venv ]; then
                uv venv .venv --python ${python}/bin/python3
              fi
              source .venv/bin/activate
            '';
          };
        });
    };
}
