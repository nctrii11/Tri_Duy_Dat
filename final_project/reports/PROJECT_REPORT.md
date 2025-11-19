## Báo cáo Tổng Kết Dự Án VN30 Markowitz

### 1. Mục tiêu & Phạm vi

- **Định nghĩa**: Xây dựng chuỗi pipeline hoàn chỉnh để thu thập, làm sạch, phân tích và tối ưu hóa danh mục VN30 theo lý thuyết Markowitz.
- **Vũ trụ**: 30 mã cố định trong rổ VN30 (từ file `configs/vn30_fixed_list.yaml`), dữ liệu gốc từ `vnstock`.
- **Mốc thời gian**: 11/06/2021 – 30/10/2025 (1 096 phiên giao dịch khả dụng sau khi giao cắt dữ liệu).
- **Kết quả**: Bộ dữ liệu sạch dưới `data/processed/`, bộ biểu đồ EDA trong `reports/eda/`, và nhật ký/log đầy đủ phục vụ kiểm toán.

### 2. Dữ liệu & Đặc tính

| Thuộc tính        | Giá trị                                                      |
| ----------------- | ------------------------------------------------------------ |
| Nguồn             | `vnstock` (API Quote)                                        |
| File đầu ra       | `data/raw/parquet/prices.parquet`, `data/raw/csv/prices.csv` |
| Số mã             | 30                                                           |
| Số ngày sau align | 1 096                                                        |
| Khoảng ngày       | 2021-06-11 → 2025-10-30                                      |

Chi tiết nằm trong log `reports/logs/fetch_data_experiment_vn30_markowitz.json`.

### 3. Pipeline kỹ thuật

| Bước               | Lệnh Hydra                                                                                           | Mô tả                                                                                                                                           |
| ------------------ | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Fetch           | `uv run python -m src.vn50.cli.fetch_data experiment=experiment_vn30_markowitz`                      | Cào dữ liệu daily adjusted close từ vnstock, lưu raw CSV/Parquet.                                                                               |
| 2. Preprocess      | `uv run python -m src.vn50.cli.preprocess_data experiment=experiment_vn30_markowitz`                 | Làm sạch, align lịch HOSE, loại giá ≤0, winsorize (rolling quantile 252 ngày), stale-mask ≥3 phiên, sinh log return ngày & simple return tháng. |
| 3. Returns CLI     | `uv run python -m src.vn50.cli.make_returns +experiment=experiment_vn30_markowitz`                   | (Nếu cần) tái tính bộ returns ở nhiều tần suất.                                                                                                 |
| 4. Optimize Manual | `uv run python -m src.vn50.cli.optimize_markowitz_manual +experiment=experiment_vn30_markowitz`      | Giải GMV/Tangency bằng scipy/cvxpy với constraint long-only.                                                                                    |
| 5. Optimize PyPO   | `uv run python -m src.vn50.cli.optimize_markowitz_pypfopt +experiment=experiment_vn30_markowitz`     | Đối chiếu kết quả thông qua PyPortfolioOpt.                                                                                                     |
| 6. Backtest        | `uv run python -m src.vn50.cli.backtest +experiment=experiment_vn30_markowitz`                       | Walk-forward, tái cân bằng hàng tháng, xuất NAV/metrics.                                                                                        |
| 7. EDA             | `uv run python -c "from src.vn50.eda.eda_plots import run_all_eda; run_all_eda()"`                   | Sinh 5 biểu đồ chính (giá, normalized, rolling vol, histogram monthly, scree).                                                                  |
| 7b. EDA Legacy     | `uv run python -c "from src.vn50.eda.eda_plots import run_requested_charts; run_requested_charts()"` | Sinh thêm 3 biểu đồ đặc thù: monthly combo, actual price overlay, heatmap tương quan tháng.                                                     |

### 4. Tiền xử lý & Chất lượng dữ liệu

Tóm tắt từ `reports/logs/preprocess_experiment_vn30_markowitz_20251118_182828.json`:

- **Số ngày sạch**: 1 096; **% ngày toàn bộ ticker thiếu**: 0%.
- **Winsorization**: Rolling quantile 0.5% & 99.5% (252 ngày), abs cap 9.5%.
- **Stale masking**: ≥3 phiên zero-return liên tiếp ⇒ 0.94% điểm bị mask.
- **Tỉ lệ returns bị clip**: 2.08%; giá trị trung bình |r| giảm từ 1.58% → 1.57%.
- **Tham số khác**: tự động loại giá ≤0, loại ngày toàn NaN, không forward-fill.

### 5. EDA & Tài liệu hình ảnh

Các biểu đồ full-HD (40×16 in, DPI 300) nằm trong `reports/eda/`:

1. `prices_actual_vn30.png`: Giá đóng cửa thực tế của 30 mã.
2. `price_history_normalized.png`: Chuỗi giá chuẩn hóa về 100 + đường bình quân VN30.
3. `rolling_volatility.png`: Volatility rolling 63 ngày (annualized) cho từng mã + trung bình.
4. `monthly_returns_hist.png`: Histogram monthly returns hợp nhất (mean & zero).
5. `covariance_eigen_analysis.png`: Scree plot & explained variance của ma trận covariance.
6. `monthly_returns_distribution.png`: Combo histogram + boxplot 15 mã đầu (legacy style).
7. `corr_heatmap_vn30_monthly.png`: Heatmap tương quan monthly (Markowitz input).

### 6. Kết quả tối ưu hóa & Backtest

Số liệu trong `reports/logs/metrics_walkforward_markowitz.json` (mốc 11/2021–10/2025):

| Chiến lược             | Ann. Return | Ann. Vol | Sharpe | Max DD | Calmar | Sortino | Turnover |
| ---------------------- | ----------- | -------- | ------ | ------ | ------ | ------- | -------- |
| GMV Manual             | 7.97%       | 11.42%   | 0.44   | -5.58% | 1.43   | 0.87    | 0.55     |
| Tangency Manual        | 8.59%       | 14.30%   | 0.39   | -9.92% | 0.87   | 0.90    | 1.05     |
| Equal-Weight Benchmark | 19.54%      | 14.62%   | 1.13   | -4.84% | 4.03   | 3.05    | 0.00     |

Kết luận sơ bộ: Equal-weight trong giai đoạn này vượt trội về Sharpe/Calmar do chu kỳ tăng hậu 2024, tuy nhiên GMV/Tangency giữ drawdown thấp và đóng vai trò kiểm soát rủi ro. Cần tiếp tục tinh chỉnh ước lượng µ–Σ (shrinkage mạnh hơn, rolling window dài hơn) để cải thiện Sharpe.

### 7. Hướng dẫn tái chạy toàn bộ

```bash
# 0. Thiết lập môi trường
uv venv --python 3.11 && source .venv/bin/activate
uv sync

# 1. Fetch + Preprocess
uv run python -m src.vn50.cli.fetch_data experiment=experiment_vn30_markowitz
uv run python -m src.vn50.cli.preprocess_data experiment=experiment_vn30_markowitz

# 2. EDA
uv run python -c "from src.vn50.eda.eda_plots import run_all_eda; run_all_eda()"

# 3. Optimize + Backtest
uv run python -m src.vn50.cli.optimize_markowitz_manual +experiment=experiment_vn30_markowitz
uv run python -m src.vn50.cli.optimize_markowitz_pypfopt +experiment=experiment_vn30_markowitz
uv run python -m src.vn50.cli.backtest +experiment=experiment_vn30_markowitz
```

### 8. Công việc tiếp theo

1. **Data**: Bổ sung calendar HOSE chuẩn để tránh cảnh báo thiếu file (`data/cache/hose_trading_days.csv`).
2. **Estimators**: Thử Ledoit-Wolf / Oracle Approximating shrinkage cho covariance, hoặc integrate Bayesian mean estimator để ổn định tangency weights.
3. **Transaction Cost Model**: Hiện backtest mới dùng chi phí mặc định; cần tham số hóa theo basis points trong config `backtest_default.yaml`.
4. **Visualization**: Tích hợp các biểu đồ mới vào chương 4 báo cáo Markowitz (LaTeX hoặc Word) qua đường dẫn `reports/eda/*.png`.
5. **Automation**: Viết script `make report` gom toàn bộ bước fetch → backtest → export log để thuận tiện khi cập nhật dữ liệu tương lai.

---

**Trạng thái**: Pipeline hoạt động end-to-end; dữ liệu sạch và log đã versioned. Người dùng có thể tái chạy hoặc mở rộng nghiên cứu (ví dụ sensitivity analysis hay shock stress-test) dựa trên cấu trúc hiện có.
