def is_estimator(model):
    try:
        return callable(getattr(model, "fit"))
    except:
        return False
