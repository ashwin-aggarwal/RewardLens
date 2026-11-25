"""Command-line handling for scripts that take no options, or only on/off flags.

Without a parser, `python -m scripts.<name> --help` (or a mistyped option) would start the full run.
"""

from __future__ import annotations

import argparse


def script_args(doc: str | None, *flags: str, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line of a script whose only options are the given on/off flags.

    `--help` prints the script's docstring and exits; any unknown argument is an error. Neither starts the run.
    """
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in flags:
        parser.add_argument(flag, action="store_true")
    return parser.parse_args(argv)
