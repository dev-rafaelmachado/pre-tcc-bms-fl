from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def pretrain_random_forest(X, y):
    """Pré-treina um modelo floresta aleatória nos dados."""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    preds = rf.predict(X)
    residuals = y - preds
    print("Random Forest training completed")
    return preds, residuals


def pretrain_decision_tree(X, y):
    """Pré-treina um modelo árvore de decisão nos dados."""
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X, y)
    preds = dt.predict(X)
    residuals = y - preds
    print("Decision Tree training completed")
    return preds, residuals
