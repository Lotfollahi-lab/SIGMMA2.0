"""Drop-in `python -m eva` replacement that pre-registers `local/uni`.

Importing this module triggers `_register_local_uni` which adds `local/uni`
to EVA's `backbone_registry`. Then we invoke EVA's CLI exactly as
`python -m eva` would.

Usage:
    python -m scripts.eva_critique._eva_cli_with_local_uni fit --config <yaml>
"""

# Side-effect: registers `local/uni` in backbone_registry.
import scripts.eva_critique._register_local_uni  # noqa: F401

from eva.__main__ import main as eva_main


if __name__ == "__main__":
    # `__main__` guard so DataLoader workers (spawned via multiprocessing)
    # can re-import this module without re-running the CLI.
    eva_main()
