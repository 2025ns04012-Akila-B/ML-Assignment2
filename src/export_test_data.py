"""Export the stratified hold-out split to ``test_data.csv``.

Run from the project root::

    python -m src.export_test_data
"""

from __future__ import annotations

from src.preprocessing import clean_dataset, split_data
from src.train_models import export_test_data
from src.utils import get_logger, load_raw_dataset

LOGGER = get_logger(__name__)


def main() -> None:
    """Recreate the (deterministic) test split and save it as CSV."""
    cleaned = clean_dataset(load_raw_dataset())
    _, test_df = split_data(cleaned)
    export_test_data(test_df)


if __name__ == "__main__":
    main()
