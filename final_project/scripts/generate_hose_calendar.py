"""Script to generate HOSE trading calendar."""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vn50.data import calendar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate HOSE trading calendar."""
    # Default date range (có thể override bằng CLI args)
    start = "2020-10-31"
    end = "2025-10-31"

    logger.info(f"Generating HOSE trading calendar from {start} to {end}")

    # Option 1: Generate từ logic đơn giản (chỉ loại cuối tuần)
    calendar_df = calendar.generate_hose_calendar(start=start, end=end)

    # Option 2: Nếu có dữ liệu giá, có thể generate từ đó
    # prices_path = Path("data/raw/prices.csv")
    # if prices_path.exists():
    #     prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    #     calendar_df = calendar.generate_calendar_from_prices(prices)

    # Save to cache
    output_path = Path("data/cache/hose_trading_days.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calendar_df.to_csv(output_path, index=False)

    logger.info(f"Saved {len(calendar_df)} trading days to {output_path}")
    logger.info(
        "Note: This is a basic calendar (weekdays only). "
        "For production, integrate with HOSE API or historical data to exclude holidays."
    )


if __name__ == "__main__":
    main()

