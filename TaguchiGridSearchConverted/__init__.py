
import sys
import fire

from .TaguchiGridSearchConverted import TaguchiGridSearchConverter

__all__ = ["TaguchiGridSearchConverter"]

__VERSION__ = TaguchiGridSearchConverter.__VERSION__

def cli_launcher() -> None:
    if sys.argv[-1] ==  "--version":
        return(f"TaguchiGridSearchConverted version: {__VERSION__}")
    fire.Fire(TaguchiGridSearchConverter)

if __name__ == "__main__":
    cli_launcher()