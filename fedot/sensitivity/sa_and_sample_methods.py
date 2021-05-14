from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample import saltelli


def sobol_method(problem: dict, samples, operation_response) -> dict:
    indices = sobol_analyze(problem, operation_response, print_to_console=False)

    return indices


def make_saltelly_sample(problem: dict, num_of_samples=100):
    samples = saltelli.sample(problem, num_of_samples)

    return samples


analyze_method_by_name = {
    'sobol': sobol_method,
}

sample_method_by_name = {
    'saltelli': make_saltelly_sample

}
