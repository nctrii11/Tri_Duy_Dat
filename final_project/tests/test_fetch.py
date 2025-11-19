import pandas as pd

from src.vn50.data import fetch


def test_fetch_prices_aligns_index_and_columns(monkeypatch):
    base_index = pd.date_range("2020-10-30", periods=4, freq="D")

    def fake_download(symbol: str, start: str, end: str, source: str) -> pd.DataFrame:
        if symbol == "BBB":
            idx = base_index[1:]
        else:
            idx = base_index[:-1]
        return pd.DataFrame(
            {
                "date": idx,
                "adjust_price": range(len(idx)),
                "close": range(len(idx)),
            }
        )

    monkeypatch.setattr(fetch, "_download_symbol_history", fake_download)

    df = fetch.fetch_prices(["AAA", "BBB"], "2020-10-30", "2025-10-30")

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert list(df.columns) == ["AAA", "BBB"]
    # Intersected index should match overlapping dates
    assert df.index.min() == pd.Timestamp("2020-10-31")
    assert df.index.max() == pd.Timestamp("2020-11-01")

