# VN30 Markowitz Portfolio Optimization

Dự án tối ưu hóa danh mục đầu tư theo mô hình Markowitz (mean-variance) cho rổ VN30 trên thị trường chứng khoán Việt Nam (HOSE).

## Mục tiêu

- Tiền xử lý & chuẩn hóa dữ liệu VN30 (không look-ahead, winsorization, stale-masking)
- Tối ưu danh mục bằng Markowitz với:
  - **Phần A**: Code tự lập trình (scipy/cvxpy)
  - **Phần B**: Thư viện PyPortfolioOpt
- Backtest trên vũ trụ 30 mã (walk-forward)
- Ghi log đầy đủ vào `reports/logs/`

## Tech Stack

- Python 3.11
- uv (package manager)
- Hydra/OmegaConf (config management)
- pandas/numpy/scipy
- PyPortfolioOpt/CVXPY
- pytest, pre-commit (Black/Ruff)

## Setup

```bash
# Tạo virtual environment và cài đặt dependencies
make setup

# Hoặc thủ công:
uv venv --python 3.11
uv sync
```

## Pipeline CLI

### Data & Preprocessing (1-3)

```bash
# 1. Đọc dữ liệu từ file local (đặt file prices.csv vào data/raw/)
uv run python -m src.vn50.cli.fetch_data +experiment=experiment_vn30_markowitz

# 2. Tiền xử lý
uv run python -m src.vn50.cli.preprocess_data +experiment=experiment_vn30_markowitz

# 3. Tính returns
uv run python -m src.vn50.cli.make_returns +experiment=experiment_vn30_markowitz
```

### Optimize & Backtest (4-6)

```bash
# 4. Tối ưu Markowitz (Manual)
uv run python -m src.vn50.cli.optimize_markowitz_manual +experiment=experiment_vn30_markowitz

# 5. Tối ưu Markowitz (PyPortfolioOpt)
uv run python -m src.vn50.cli.optimize_markowitz_pypfopt +experiment=experiment_vn30_markowitz

# 6. Backtest
uv run python -m src.vn50.cli.backtest +experiment=experiment_vn30_markowitz
```

## Testing & Linting

```bash
# Chạy tests
make test

# Lint code
make lint
```

## Cấu trúc dự án

```
.
├── configs/              # Hydra configs
│   ├── data/
│   ├── preprocess/
│   ├── features/
│   ├── split/
│   ├── markowitz_manual/
│   ├── markowitz_pypfopt/
│   ├── backtest/
│   ├── paths/
│   └── calendar/
├── src/vn50/
│   ├── data/             # Fetch & preprocess
│   ├── features/         # Returns, EDA
│   ├── split/            # Time split
│   ├── markowitz/        # Estimators
│   ├── optimize/         # Markowitz optimization
│   ├── backtest/         # Backtest engine & metrics
│   └── cli/              # CLI scripts
├── data/                 # Raw/interim/processed data
├── reports/              # Logs, figures, artifacts
└── tests/                # Unit tests
```

## Benchmark

- VNINDEX
- VN30 Index

## Thời gian dữ liệu

- Từ 2020-10-31 đến 2025-10-31 (~5 năm giao dịch)
- In-sample: ~3 năm đầu
- Out-of-sample: ~2 năm cuối

