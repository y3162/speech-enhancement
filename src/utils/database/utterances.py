import argparse

from .corpora.dto import UTTERANCES_TABLE
from .corpora import (
    librispeech_utterance_iterator,
    libritts_utterance_iterator,
    vctk_utterance_iterator,
)
from .common import import_rows
from .constants import SPEECH_UTILS_DB_METADATA_PATH


CORPUS_ITERATORS = {
    "librispeech": librispeech_utterance_iterator,
    "libritts": libritts_utterance_iterator,
    "vctk": vctk_utterance_iterator,
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
        UTTERANCES_TABLE,
        list(iterator()),
        force=args.force,
    )
