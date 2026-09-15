import argparse

from .corpora.dto import (
    NOISES_TABLE,
    Noise,
)
from .corpora import (
    demand_noise_iterator,
)
from .connection import Connection
from .common import (
    create_database,
    create_table,
)
from .config import (
    SPEECH_UTILS_DB_METADATA_PATH,
)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--corpus", type=str, required=True)
    args.add_argument("--force", action="store_true")
    args = args.parse_args()

    match args.corpus:
        case "demand":
            noise_generator = demand_noise_iterator()
        case _:
            raise NotImplementedError(f"Corpus {args.corpus} not implemented")

    create_database(SPEECH_UTILS_DB_METADATA_PATH, force=args.force)
    create_table(
        SPEECH_UTILS_DB_METADATA_PATH,
        NOISES_TABLE,
    )

    BATCH_SIZE = 10_000

    con = Connection(SPEECH_UTILS_DB_METADATA_PATH)
    batch: list[Noise] = []
    for noise in noise_generator:
        batch.append(noise)
        if len(batch) < BATCH_SIZE:
            continue
        con.insert(NOISES_TABLE, batch)
        batch = []
    if batch:
        con.insert(NOISES_TABLE, batch)
    con.commit()
    con.close()
