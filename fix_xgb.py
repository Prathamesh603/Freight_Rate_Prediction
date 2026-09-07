import json

with open('D:/Assignments/Spotter.AI/freight_rate_prediction.ipynb', 'r') as f:
    nb = json.load(f)

old_str = 'xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)'
new_str = 'xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[xgb.callback.EarlyStopping(rounds=50, metric_name="rmse")])'

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if isinstance(cell['source'], list):
            cell['source'] = [line.replace(old_str, new_str) for line in cell['source']]
        else:
            cell['source'] = cell['source'].replace(old_str, new_str)

with open('D:/Assignments/Spotter.AI/freight_rate_prediction.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Fixed XGBoost early_stopping_rounds')
