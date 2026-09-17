from pathlib import Path

from src.se.training.loop import main

if __name__ == "__main__":
    main(
        name="se_mamba_pp",
        default_config=Path(__file__).parent / "configs" / "default.json",
        description="Train SEMamba++",
    )
