from fedot.api.main import Fedot
import numpy as np
import pandas as pd
import string

import cProfile
import pstats
from pstats import SortKey

df = pd.DataFrame(np.random.randint(0, 100, size=(10, 10)), columns=list(string.ascii_lowercase[0:10:1]))

auto_model = Fedot(
    problem="regression",
    metric=["rmse"],
    preset="best_quality",
    with_tuning=True,
    timeout=1,
    cv_folds=5,
    seed=42,
    n_jobs=1,
    # logging_level=10,
    use_pipelines_cache=True,
    use_auto_preprocessing=False,
)

auto_model.fit(features=df, target="a")
# cProfile.run('auto_model.fit(features=df, target="a")', "restats")

# p = pstats.Stats("restats")
# p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(".*cache.*")

# prediction = auto_model.predict(features=test, save_predictions=True)

print(auto_model.get_metrics())
print(auto_model.return_report().head(10))
