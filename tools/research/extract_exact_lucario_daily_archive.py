"""Stream an official daily archive and extract exact-77a Lucario games.

This is a thin target-specific binding over the already-audited streaming
extractor.  It does not expand the ZIP wholesale and writes only games with at
least one seat using the exact Hariyama/Mega-Lucario registration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import extract_exact_dragapult_daily_archive as BASE  # noqa: E402


TARGET_DECK = tuple(sorted((
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    673, 673, 674, 674, 675, 675, 676, 676, 676,
    677, 677, 677, 678, 678, 678, 678,
    1121, 1121, 1121, 1121, 1123, 1123,
    1141, 1141, 1141, 1141, 1142, 1142, 1142, 1142,
    1152, 1152, 1152, 1152, 1159, 1182, 1182,
    1213, 1213, 1213, 1213, 1227, 1227, 1227, 1227,
    1229, 1229,
)))
TARGET_SHA256 = "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8"


class _LucarioTarget:
    TARGET_DECK = TARGET_DECK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-dir", action="append", type=Path, default=[])
    args = parser.parse_args()
    if BASE.deck_sha(TARGET_DECK) != TARGET_SHA256:
        parser.error("bound Lucario deck hash drifted")
    # The base function reads these two globals only to bind target/schema.
    # Its implementation remains byte-identical to the prior Dragapult run.
    original_target, original_schema = BASE.DRAGAPULT, BASE.SCHEMA
    BASE.DRAGAPULT = _LucarioTarget
    BASE.SCHEMA = "ptcg.lucario.daily-archive-extraction.v1"
    try:
        result = BASE.extract(args.archive, args.out, args.skip_dir)
    except (BASE.ExtractionError, OSError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    finally:
        BASE.DRAGAPULT, BASE.SCHEMA = original_target, original_schema
    print(json.dumps({
        "archive_sha256": result["archive"]["sha256"],
        "target_deck_sha256": result["target_deck_sha256"],
        "counters": result["counters"],
        "teachers": result["teachers"],
        "manifest_sha256": result["manifest_sha256"],
    }, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
