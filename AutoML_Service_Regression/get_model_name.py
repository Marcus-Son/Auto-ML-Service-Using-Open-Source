def get_model_name(model):
    model_name_new = None
    if isinstance(model,str):
        if model == "Linear Regression":
            model_name_new = "Linear Regression"
        elif model == "Ridge Regression":
            model_name_new = "Ridge Regression"
        elif model == "Lasso Regression":
            model_name_new = "Lasso Regression"
        elif model == "Elastic Net":
            model_name_new = "Elastic Net"
        elif model == "Decision Tree Regressor":
            model_name_new = "Decision Tree Regressor"
        elif model == "Random Forest Regressor":
            model_name_new = "Random Forest Regressor"
        elif model == "Extra Trees Regressor":
            model_name_new = "Extra Trees Regressor"
    else:
        model_type = type(model).__name__
        if model_type == "LinearRegression":
            model_name_new = "Linear Regression"
        elif model_type == "Ridge":
            model_name_new = "Ridge Regression"
        elif model_type == "Lasso":
            model_name_new = "Lasso Regression"
        elif model_type == "ElasticNet":
            model_name_new = "Elastic Net"
        elif model_type == "DecisionTreeRegressor":
            model_name_new = "Decision Tree Regressor"
        elif model_type == "RandomForestRegressor":
            model_name_new = "Random Forest Regressor"
        elif model_type == "ExtraTreesRegressor":
            model_name_new = "Extra Trees Regressor"

    return model_name_new
