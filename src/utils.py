import os
import sys

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

if not hasattr(np, "_core") and hasattr(np, "core"):
    sys.modules["numpy._core"] = np.core
    if hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    if hasattr(np.core, "_multiarray_umath"):
        sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        try:
            from sklearn.impute import SimpleImputer

            def _patch_imputer(item):
                if isinstance(item, SimpleImputer) and not hasattr(item, "_fill_dtype"):
                    item._fill_dtype = getattr(item.statistics_, "dtype", np.dtype("float64"))

            if hasattr(obj, "transformers_"):
                for _, transformer, _ in obj.transformers_:
                    if hasattr(transformer, "steps"):
                        for _, step in transformer.steps:
                            _patch_imputer(step)
                    else:
                        _patch_imputer(transformer)
            elif hasattr(obj, "steps"):
                for _, step in obj.steps:
                    _patch_imputer(step)
            _patch_imputer(obj)
        except Exception:
            pass

        return obj

    except Exception as e:
        raise CustomException(e, sys)

