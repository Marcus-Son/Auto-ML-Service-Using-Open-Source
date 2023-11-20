def best_model(best_model):
    best_model = best_model.strip()
    if best_model == "Linear Regression":
        best_model = "lr"
    elif best_model == "Ridge Regression":
        best_model = "ridge"
    elif best_model == "Lasso Regression":
        best_model = "lasso"
    elif best_model == "Elastic Net":
        best_model = "en"
    elif best_model == "Decision Tree Regressor":
        best_model = "dt"
    elif best_model == "Random Forest Regressor":
        best_model = "rf"
    elif best_model == "Extra Trees Regressor":
        best_model = "et"

    return best_model


