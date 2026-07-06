from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
def get_model(model_name, X_train, y_train):

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)

    elif model_name == "XGBoost":
        model = XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )

    elif model_name == "Random Forest":
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None]
        }

        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring="accuracy"
        )

        grid.fit(X_train, y_train)

        model = grid.best_estimator_

    if model_name != "Random Forest":
        model.fit(X_train, y_train)

    return model