from datetime import datetime
import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(curdir, '..'))
ROOT = os.path.abspath(os.curdir)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "fedot"))

from cuml.svm import SVC as gpu_svc
from sklearn.datasets import make_moons
from sklearn.svm import SVC as cpu_svc


def run_gpu_svc(n_samples, X, y):
    model = gpu_svc(kernel='rbf', C=10, gamma=1, cache_size=2000)
    start = datetime.now()
    model.fit(X, y)
    print(f'finished gpu {n_samples} in {datetime.now() - start}')
    y_pred = model.predict(X)


def run_cpu_svc(n_samples, X, y):
    model = cpu_svc(kernel='rbf', C=10, gamma=1, cache_size=2000)
    start = datetime.now()
    model.fit(X, y)
    print(f'finished cpu {n_samples} in {datetime.now() - start}')
    y_pred = model.predict(X)


if __name__ == '__main__':
    n_samples = [10000, 100000, 200000, 300000]
    for sample in n_samples:
        X, y = make_moons(n_samples=sample, shuffle=True, noise=0.1, random_state=137)
        run_gpu_svc(sample, X, y)
        run_cpu_svc(sample, X, y)
