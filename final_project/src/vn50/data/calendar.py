"""Generate HOSE trading calendar."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def generate_hose_calendar(start: str, end: str) -> pd.DataFrame:
    """
    Generate HOSE trading calendar for date range [start, end].

    Trả về DataFrame có cột 'date' là toàn bộ ngày giao dịch HOSE
    trong khoảng [start, end].

    Lưu ý: Hàm này chỉ là skeleton. Để có dữ liệu thực tế, cần:
    1. Kết nối với API HOSE hoặc nguồn dữ liệu đáng tin cậy
       (vnstock, SSI, VNDirect)
    2. Hoặc extract từ dữ liệu lịch sử giá
       (ngày nào có giá = ngày giao dịch)
    3. Loại bỏ ngày nghỉ lễ, cuối tuần

    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)

    Returns:
        DataFrame with column 'date' containing trading days
    """
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    # TODO: Thay thế logic này bằng fetch thực tế từ API/nguồn dữ liệu
    # Hiện tại: tạo tất cả ngày trong khoảng
    # (bao gồm cả cuối tuần, chưa loại lễ)
    # Trong thực tế, cần:
    # - Loại bỏ thứ 7, Chủ nhật
    # - Loại bỏ ngày nghỉ lễ Việt Nam (Tết, 30/4, 1/5, 2/9, v.v.)
    # - Loại bỏ ngày tạm ngừng giao dịch của HOSE

    all_days = pd.date_range(start=start_date, end=end_date, freq="D")
    # Loại bỏ cuối tuần (thứ 7 = 5, Chủ nhật = 6)
    trading_days = all_days[all_days.weekday < 5]

    calendar_df = pd.DataFrame({"date": trading_days})

    logger.warning(
        "Generated calendar using weekday filter only. "
        "TODO: Integrate with HOSE API or historical data "
        "to exclude holidays."
    )

    return calendar_df


def generate_calendar_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading calendar from existing price data.

    Ngày nào có ít nhất một mã có giá (không phải NaN) = ngày giao dịch.

    Args:
        prices: DataFrame with DatetimeIndex, columns are tickers

    Returns:
        DataFrame with column 'date' containing trading days
    """
    # Ngày nào có ít nhất một giá trị không NaN
    has_data = ~prices.isna().all(axis=1)
    trading_days = prices.index[has_data]

    calendar_df = pd.DataFrame({"date": trading_days})

    logger.info(
        f"Generated {len(trading_days)} trading days from price data "
        f"({trading_days[0].date()} to {trading_days[-1].date()})"
    )

    return calendar_df
