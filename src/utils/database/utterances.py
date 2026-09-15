from .corpora.dto import UTTERANCES_TABLE
from .common import (
    create_database,
    create_table,
)
from .config import (
    SPEECH_UTILS_DB_METADATA_PATH,
)


def create_utterances_table() -> None:
    create_database(SPEECH_UTILS_DB_METADATA_PATH)
    create_table(
        SPEECH_UTILS_DB_METADATA_PATH,
        UTTERANCES_TABLE,
    )
