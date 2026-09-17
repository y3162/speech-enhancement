import argparse

from .corpora.librispeech import utterance_iterator as librispeech_utterance_iterator
from .corpora.libritts import utterance_iterator as libritts_utterance_iterator
from .corpora.vctk import utterance_iterator as vctk_utterance_iterator
from .db import import_rows, metadata_path
from .schema import UTTERANCES_TABLE

CORPUS_ITERATORS = {
    "librispeech": librispeech_utterance_iterator,
    "libritts": libritts_utterance_iterator,
    "vctk": vctk_utterance_iterator,
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
        UTTERANCES_TABLE,
        rows,
        replace=args.replace,
    )
