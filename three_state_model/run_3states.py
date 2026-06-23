from src.data_loader import download_and_process_data
from src.performance_metrics import calculate_metrics
from three_state_model.backtest_3states import TRAIN_WINDOW, run_backtest_3states


def main():
    market_data = download_and_process_data()
    strategy_results = run_backtest_3states(market_data, train_window=TRAIN_WINDOW)
    calculate_metrics(strategy_results)


if __name__ == "__main__":
    main()
