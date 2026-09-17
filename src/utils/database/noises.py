import argparse

from .common import import_rows
from .constants import SPEECH_UTILS_DB_METADATA_PATH
from .corpora import (
    demand_noise_iterator,
)
from .corpora.dto import NOISES_TABLE

CORPUS_ITERATORS = {
    "demand": demand_noise_iterator,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    iterator = CORPUS_ITERATORS.get(args.corpus)
    if iterator is None:
        raise NotImplementedError(f"Corpus {args.corpus} not implemented")

    import_rows(
        SPEECH_UTILS_DB_METADATA_PATH,
        NOISES_TABLE,
        list(iterator()),
        force=args.force,
    )
