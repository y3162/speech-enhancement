import argparse

from .corpora.dto import (
    UTTERANCES_TABLE,
    Utterance,
)
from .corpora import (
    librispeech_utterance_iterator,
    libritts_utterance_iterator,
    vctk_utterance_iterator,
)
from .connection import Connection
from .common import (
    create_database,
    create_table,
)
from .constants import (
    SPEECH_UTILS_DB_METADATA_PATH,
)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--corpus", type=str, required=True)
    args.add_argument("--force", action="store_true")
    args = args.parse_args()

    match args.corpus:
        case "librispeech":
            utterance_generator = librispeech_utterance_iterator()
        case "libritts":
            utterance_generator = libritts_utterance_iterator()
        case "vctk":
            utterance_generator = vctk_utterance_iterator()
        case _:
            raise NotImplementedError(f"Corpus {args.corpus} not implemented")

    create_database(SPEECH_UTILS_DB_METADATA_PATH, force=args.force)
    create_table(
        SPEECH_UTILS_DB_METADATA_PATH,
        UTTERANCES_TABLE,
    )

    BATCH_SIZE = 10_000

    con = Connection(SPEECH_UTILS_DB_METADATA_PATH)
    batch: list[Utterance] = []
    for utterance in utterance_generator:
        batch.append(utterance)
        if len(batch) < BATCH_SIZE:
            continue
        con.insert(UTTERANCES_TABLE, batch)
        batch = []
    if batch:
        con.insert(UTTERANCES_TABLE, batch)
    con.commit()
    con.close()
