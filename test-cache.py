from fedot.api.main import Fedot
import numpy as np
import pandas as pd
import string

import cProfile
import pstats
from pstats import SortKey

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

df = pd.DataFrame(np.random.randint(0, 100, size=(10, 10)), columns=list(string.ascii_lowercase[0:10:1]))

X, y = make_regression(n_samples=100, n_features=1, noise=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

auto_model = Fedot(
    problem="regression",
    metric=["mae", "mse", "r2"],
    preset="best_quality",
    with_tuning=False,
    timeout=2.5,
    cv_folds=5,
    seed=42,
    n_jobs=1,
    # logging_level=10,
    use_pipelines_cache=False,
    use_auto_preprocessing=False,
)

auto_model.fit(features=X_train, target=y_train)
# cProfile.run('auto_model.fit(features=df, target="a")', "restats")

# p = pstats.Stats("restats")
# p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(".*cache.*")

prediction = auto_model.predict(features=X_test, save_predictions=False)

print(auto_model.get_metrics())
print(auto_model.return_report().head(10))