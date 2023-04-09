from sklearn.metrics import r2_score, mean_squared_error

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)