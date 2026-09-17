from pathlib import Path

from src.se.training.loop import main

if __name__ == "__main__":
    main(
        name="mp_senet",
        default_config=Path(__file__).parent / "configs" / "transformer.json",
        description="Train MP-SENet",
    )
