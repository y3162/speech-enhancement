import argparse

from .corpora.demand import noise_iterator as demand_noise_iterator
from .db import import_rows, metadata_path
from .schema import NOISES_TABLE

CORPUS_ITERATORS = {
    "demand": demand_noise_iterator,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    iterator = CORPUS_ITERATORS.get(args.corpus)
    if iterator is None:
        raise NotImplementedError(f"Corpus {args.corpus} not implemented")

    rows = sorted(iterator(), key=lambda row: row.audio_path.as_posix())
    import_rows(
        metadata_path(),
        NOISES_TABLE,
        rows,
        replace=args.replace,
    )
