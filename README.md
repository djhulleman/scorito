# WK 2026 Scorito App

Mobile-friendly Streamlit app for the existing World Cup forecasting outputs.

## Run Locally

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m streamlit run app.py
```

The app reads the generated CSVs in the `outputs*` folders. Refresh model data with the individual forecasting scripts before launching when you want newer predictions.

## Streamlit Cloud

1. Push this folder to a GitHub repository.
2. Create a new Streamlit Cloud app.
3. Set the entry point to `app.py`.
4. Keep `predictions.json` and `results_cache.json` tracked if you want picks and the latest result cache to survive redeploys.

## State Files

`predictions.json` stores top-scorer and bracket picks. `results_cache.json` stores the latest schedule/result cache and is refreshed by the app at most once per hour.
