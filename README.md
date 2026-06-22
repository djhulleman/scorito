# WK 2026 Scorito App

Mobile-friendly Streamlit app for the existing World Cup forecasting outputs.

## Run Locally

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m streamlit run app.py
```

The app reads the generated CSVs in the `outputs*` folders. Refresh model data with the individual forecasting scripts before launching when you want newer predictions.
When deployed, the app also checks for fresh results and can regenerate model outputs automatically.

## Streamlit Cloud

1. Push this folder to a GitHub repository.
2. Create a new Streamlit Cloud app.
3. Set the entry point to `app.py`.
4. Keep `predictions.json` and `results_cache.json` tracked if you want picks and the latest result cache to survive redeploys.

## State Files

`predictions.json` stores top-scorer and bracket picks. `results_cache.json` stores the latest schedule/result cache and is refreshed by the app at most once per hour.
`model_refresh_state.json` records when model outputs were last regenerated. The app reruns the exact-score, top-scorer, and poule scripts when either 48 hours have passed or the number of final scores changes.

## Bookmaker Benchmarks

Historical 1X2 bookmaker odds are loaded into `outputs_bookmaker_backtest/bookmaker_1x2_odds_used.csv`.
The app reads this saved CSV only; it does not fetch bookmaker odds during normal page navigation.
To rebuild the benchmark from the saved odds:

```powershell
py -3 -B bookmaker_odds_backtest.py --odds-csv outputs_bookmaker_backtest/bookmaker_1x2_odds_used.csv
```

The local `.cache_world_cup/` folder is ignored because it contains fetched HTML/API cache files.
