"""Freight rate prediction - leakage-safe temporal pipeline.
Run:  python train.py   (from the repo root)
Outputs: validation_predictions.csv, Data/december-chart-inputs.csv (filled),
         Data/december_chart_inputs.csv (same content, name used by the score example),
         outputs/feature_importance.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

BASE = Path(__file__).parent
DATA = BASE / "Data"
SEED = 42
SMOOTHING = 100.0
R_EARTH = 3958.8

# ---------------- cleaning (fit on train only) ----------------
def fit_clean_stats(train):
    wm = train.groupby("equipment")["weight"].median()
    mm = train["market_index"].median()
    return {"weight_by_equip": wm, "market_median": mm,
            "global_weight": train["weight"].median()}

def apply_clean(df, stats):
    df = df.copy()
    df["weight"] = df["weight"].abs()  # 292 train / 145 valid negatives: data-entry sign errors, magnitude valid
    wmap = stats["weight_by_equip"]
    gw = stats["global_weight"]
    df["weight"] = df.apply(lambda r: r["weight"] if pd.notna(r["weight"]) else wmap.get(r["equipment"], gw), axis=1)
    df["market_index"] = df["market_index"].fillna(stats["market_median"])
    return df

# ---------------- feature engineering (target-free, safe) ----------------
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R_EARTH * 2 * np.arcsin(np.sqrt(a))

def engineer(df):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_month"] = out["date"].dt.day
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["lat_diff"] = (out["pickup_lat"] - out["delivery_lat"]).abs()
    out["lon_diff"] = (out["pickup_lon"] - out["delivery_lon"]).abs()
    out["haversine_dist"] = haversine(out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"])
    d = out["distance"].clip(lower=1)
    h = out["haversine_dist"].clip(lower=1)
    out["road_hav_ratio"] = d / h
    out["road_hav_diff"] = d - h
    out["weight_per_mile"] = out["weight"] / d
    out["log_distance"] = np.log1p(out["distance"])
    out["log_weight"] = np.log1p(out["weight"])
    out["distance_x_weight"] = out["distance"] * out["weight"]
    out["distance_x_market"] = out["distance"] * out["market_index"]
    out["market_x_quote"] = out["market_index"] * out["quote_signal"]
    out["distance_x_quote"] = out["distance"] * out["quote_signal"]
    out["route"] = out["pickup"] + " -> " + out["delivery"]
    return out

# ---------------- leakage-safe target encoding ----------------
def fit_target_enc(train, col, target="posted_rate", smoothing=SMOOTHING):
    gm = train[target].mean()
    agg = train.groupby(col)[target].agg(["mean", "count"])
    smooth = (agg["count"] * agg["mean"] + smoothing * gm) / (agg["count"] + smoothing)
    return smooth.to_dict(), gm

def apply_target_enc(df, col, mapping, fallback, new_col):
    df[new_col] = df[col].map(mapping).fillna(fallback)
    return df

def add_encodings(train, valid, cols=("pickup", "delivery", "route")):
    train = train.copy(); valid = valid.copy()
    for c in cols:
        mp, gm = fit_target_enc(train, c)
        train = apply_target_enc(train, c, mp, gm, f"{c}_enc")
        valid = apply_target_enc(valid, c, mp, gm, f"{c}_enc")
    # route popularity (count, not target -> fit on train only, still map)
    rc = train["route"].value_counts()
    train["route_popularity"] = train["route"].map(rc)
    valid["route_popularity"] = valid["route"].map(rc).fillna(0)
    return train, valid

DROP = ["load_id", "pickup", "delivery", "route", "date", "posted_rate",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]

def to_matrix(train, valid):
    tr = pd.get_dummies(train, columns=["equipment"], drop_first=True)
    va = pd.get_dummies(valid, columns=["equipment"], drop_first=True)
    feats = [c for c in tr.columns if c not in DROP]
    va = va.reindex(columns=feats, fill_value=0)
    tr = tr.reindex(columns=feats + ["posted_rate"], fill_value=0)
    return tr[feats], tr["posted_rate"], va[feats], feats

def make_models():
    lgbm = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
                             min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                             reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1)
    xgbm = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7,
                            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                            random_state=SEED, verbosity=0,
                            early_stopping_rounds=50, eval_metric="rmse")  # 3.x: ctor, not fit()
    cbm = CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=7,
                            loss_function="RMSE", random_seed=SEED, verbose=0)
    return {"LightGBM": lgbm, "XGBoost": xgbm, "CatBoost": cbm}

def fit_eval(name, model, Xtr, ytr, Xva, yva, log_target=False):
    yt = np.log1p(ytr) if log_target else ytr
    if name == "LightGBM":
        model.fit(Xtr, yt, eval_set=[(Xva, np.log1p(yva) if log_target else yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    elif name == "XGBoost":
        model.fit(Xtr, yt, eval_set=[(Xva, np.log1p(yva) if log_target else yva)], verbose=False)
    else:
        model.fit(Xtr, yt, eval_set=(Xva, np.log1p(yva) if log_target else yva),
                  early_stopping_rounds=50)
    pred = model.predict(Xva)
    if log_target:
        pred = np.expm1(pred)
    pred = np.maximum(pred, 1.0)
    return {"MAE": mean_absolute_error(yva, pred), "RMSE": np.sqrt(mean_squared_error(yva, pred)),
            "R2": r2_score(yva, pred), "pred": pred, "model": model}

def build_fold(train_raw, va_raw):
    stats = fit_clean_stats(train_raw)
    tr = apply_clean(train_raw, stats); va = apply_clean(va_raw, stats)
    tr = engineer(tr); va = engineer(va)
    tr, va = add_encodings(tr, va)
    return to_matrix(tr, va)

def main():
    train_raw = pd.read_csv(DATA / "train-test.csv")
    valid_raw = pd.read_csv(DATA / "validation.csv")
    print(f"TRAIN {train_raw.shape} {train_raw['date'].min()}->{train_raw['date'].max()}")
    print(f"VALID {valid_raw.shape} {valid_raw['date'].min()}->{valid_raw['date'].max()}")
    print(f"target mean {train_raw['posted_rate'].mean():.2f} median {train_raw['posted_rate'].median():.2f} "
          f"std {train_raw['posted_rate'].std():.2f} skew {train_raw['posted_rate'].skew():.2f}")
    train_raw["date"] = pd.to_datetime(train_raw["date"])

    folds = {
        "PRIMARY Jan-Aug -> Sep-Oct": (train_raw[train_raw["date"] < "2025-09-01"], train_raw[train_raw["date"] >= "2025-09-01"]),
        "ROLL1 Jan-Jun -> Jul": (train_raw[train_raw["date"] < "2025-07-01"], train_raw[(train_raw["date"] >= "2025-07-01") & (train_raw["date"] < "2025-08-01")]),
        "ROLL2 Jan-Jul -> Aug": (train_raw[train_raw["date"] < "2025-08-01"], train_raw[(train_raw["date"] >= "2025-08-01") & (train_raw["date"] < "2025-09-01")]),
    }
    for k, (a, b) in folds.items():
        print(f"{k}: train {len(a):,} val {len(b):,}")

    # ---- primary: baselines + 3 models x raw/log ----
    tr_raw, va_raw = folds["PRIMARY Jan-Aug -> Sep-Oct"]
    Xtr, ytr, Xva, feats = build_fold(tr_raw, va_raw)
    ytr = np.asarray(ytr); yva = np.asarray(va_raw["posted_rate"].values)
    print(f"features ({len(feats)}): {feats}")

    med, mean = float(np.median(ytr)), float(np.mean(ytr))
    rpm = float(np.median(tr_raw["posted_rate"] / tr_raw["distance"].clip(lower=1)))
    va_dist = va_raw["distance"].clip(lower=1).values
    rows = []
    for bn, bp in [("Base-median", np.full_like(yva, med, dtype=float)),
                   ("Base-mean", np.full_like(yva, mean, dtype=float)),
                   ("Base-$/mile", va_dist * rpm)]:
        rows.append({"Model": bn, "Target": "raw", "MAE": mean_absolute_error(yva, bp),
                     "RMSE": np.sqrt(mean_squared_error(yva, bp)), "R2": r2_score(yva, bp)})
    results = {}
    for log in [False, True]:
        for name, m in make_models().items():
            r = fit_eval(name, m, Xtr, ytr, Xva, yva, log_target=log)
            tag = f"{name}{'-log' if log else ''}"
            results[tag] = r
            rows.append({"Model": name, "Target": "log1p" if log else "raw",
                         "MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"]})
            print(f"{tag}: MAE ${r['MAE']:.2f} RMSE ${r['RMSE']:.2f} R2 {r['R2']:.4f}")
    comp = pd.DataFrame(rows).sort_values("MAE")
    print(comp.to_string(index=False))
    best = comp[~comp["Model"].str.startswith("Base")].iloc[0]
    print(f"BEST: {best['Model']} + {best['Target']} MAE ${best['MAE']:.2f}")

    # ---- rolling stability (raw only, fewer trees for speed) ----
    print("\n--- rolling stability (raw) ---")
    for k in ["ROLL1 Jan-Jun -> Jul", "ROLL2 Jan-Jul -> Aug"]:
        a, b = folds[k]
        Xa, ya, Xb, _ = build_fold(a, b)
        yb = b["posted_rate"].values
        for name, m in make_models().items():
            if name == "LightGBM":
                m.set_params(n_estimators=800)
            elif name == "XGBoost":
                m.set_params(n_estimators=800)
            else:
                m.set_params(iterations=800)
            r = fit_eval(name, m, Xa, ya, Xb, yb, log_target=False)
            print(f"{k} {name}: MAE ${r['MAE']:.2f} RMSE ${r['RMSE']:.2f} R2 {r['R2']:.4f}")

    # ---- final training on all 48k ----
    use_log = (best["Target"] == "log1p")
    wname = best["Model"]
    stats = fit_clean_stats(train_raw)
    tr_full = engineer(apply_clean(train_raw, stats))
    va_full = engineer(apply_clean(valid_raw, stats))
    tr_full, va_full = add_encodings(tr_full, va_full)
    Xf, yf, Xvf, feats = to_matrix(tr_full, va_full)
    final = make_models()[wname]
    yt = np.log1p(yf) if use_log else yf
    final.fit(Xf, yt)
    print(f"Final {wname}{'-log' if use_log else ''} trained on {len(Xf):,}")
    vp = np.maximum(np.expm1(final.predict(Xvf)) if use_log else final.predict(Xvf), 1.0)
    out = pd.DataFrame({"load_id": valid_raw["load_id"], "predicted_rate": np.round(vp, 2)})
    out.to_csv(BASE / "validation_predictions.csv", index=False)
    print(f"Saved validation_predictions.csv {out.shape} range ${out['predicted_rate'].min():.2f}-${out['predicted_rate'].max():.2f} mean ${out['predicted_rate'].mean():.2f}")

    # ---- december: principled fallback ----
    dec_tmpl = pd.read_csv(DATA / "december-chart-inputs.csv")  # dash template
    lex = train_raw[train_raw["pickup"] == "Lexington"][["pickup_lat", "pickup_lon"]].median()
    fw = train_raw[train_raw["delivery"] == "Fort Wayne"][["delivery_lat", "delivery_lon"]].median()
    recent = train_raw[train_raw["date"] >= "2025-09-01"]
    mk_fallback = float(recent["market_index"].median())
    q_fallback = float(recent[recent["equipment"] == "Dry Van"]["quote_signal"].median())
    print(f"Dec fallback: lex {lex.to_dict()} fw {fw.to_dict()} market {mk_fallback:.4f} quote {q_fallback:.4f} (Sep-Oct trailing median, no Dec market observed)")
    dec = dec_tmpl.copy()
    dec["pickup_lat"], dec["pickup_lon"] = lex["pickup_lat"], lex["pickup_lon"]
    dec["delivery_lat"], dec["delivery_lon"] = fw["delivery_lat"], fw["delivery_lon"]
    dec["market_index"] = mk_fallback
    dec["quote_signal"] = q_fallback
    dec = engineer(dec)
    # map encodings from FULL train
    for c in ("pickup", "delivery", "route"):
        mp, gm = fit_target_enc(tr_full, c)
        dec = apply_target_enc(dec, c, mp, gm, f"{c}_enc")
    rc = tr_full["route"].value_counts()
    dec["route_popularity"] = dec["route"].map(rc).fillna(0)
    dec_m = pd.get_dummies(dec, columns=["equipment"], drop_first=True).reindex(columns=feats, fill_value=0)
    dp = np.maximum(np.expm1(final.predict(dec_m[feats])) if use_log else final.predict(dec_m[feats]), 1.0)
    dec_out = dec_tmpl.copy()
    dec_out["predicted_rate"] = np.round(dp, 2)
    dec_out.to_csv(DATA / "december-chart-inputs.csv", index=False)
    dec_out.to_csv(DATA / "december_chart_inputs.csv", index=False)  # same content, underscore name used by the score example
    print(f"Saved december-chart-inputs.csv range ${dec_out['predicted_rate'].min():.2f}-${dec_out['predicted_rate'].max():.2f}")
    print(dec_out[["date", "predicted_rate"]].to_string(index=False))

    # ---- feature importance ----
    try:
        imp = final.feature_importances_ if hasattr(final, "feature_importances_") else final.get_feature_importance()
        fi = pd.DataFrame({"feature": feats, "importance": imp}).sort_values("importance", ascending=False)
        print(fi.head(15).to_string(index=False))
        plt.figure(figsize=(10, 8))
        top = fi.head(15).iloc[::-1]
        plt.barh(top["feature"], top["importance"])
        plt.title(f"Top 15 - {wname}")
        plt.tight_layout()
        (BASE / "outputs").mkdir(exist_ok=True)
        plt.savefig(BASE / "outputs" / "feature_importance.png", dpi=150)
        print("Saved outputs/feature_importance.png")
    except Exception as e:
        print("importance skipped:", e)

if __name__ == "__main__":
    main()
