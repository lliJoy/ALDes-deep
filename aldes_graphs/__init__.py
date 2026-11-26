"""Unique import namespace for the user-supplied Algorithms directory.

Avoid collisions with other projects named Algorithms on the Python path.
Both editable installs and wheels keep Algorithms beside this package.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "Algorithms")]
