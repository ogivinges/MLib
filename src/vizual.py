import shap


def shap_by_row(model, train, df, max_display=30):
    explainer = shap.TreeExplainer(model, train, feature_perturbation='interventional')
    shap_vals = explainer(df, check_additivity=False)
    shap.plots.waterfall(shap_vals, max_display=max_display)


def shap_importance(model, train, df, max_display=30):
    explainer = shap.TreeExplainer(model, train, feature_perturbation='interventional')
    shap_vals = explainer(df, check_additivity=False)
    shap.plots.beeswarm(shap_vals, max_display=max_display)