from __future__ import division, print_function

from fedot.industrial.core.architecture.settings.computational import backend_methods as np


class RecurrenceFeatureExtractor:
    def __init__(self, recurrence_matrix: np.ndarray = None):
        self.recurrence_matrix = recurrence_matrix

    def quantification_analysis(
            self,
            MDL: int = 3,
            MVL: int = 3,
            MWVL: int = 2):

        n_vectors = self.recurrence_matrix.shape[0]
        recurrence_rate = float(
            np.sum(self.recurrence_matrix)) / np.power(n_vectors, 2)

        diagonal_frequency_dist = self.calculate_diagonal_frequency(
            number_of_vectors=n_vectors)
        vertical_frequency_dist = self.calculate_vertical_frequency(
            number_of_vectors=n_vectors, not_white=1)
        white_vertical_frequency_dist = self.calculate_vertical_frequency(
            number_of_vectors=n_vectors, not_white=0)

        determinism = self.laminarity_or_determinism(
            MDL, n_vectors, diagonal_frequency_dist, lam=False)
        laminarity = self.laminarity_or_determinism(
            MVL, n_vectors, vertical_frequency_dist, lam=True)

        average_diagonal_line_length = self.average_line_length(
            MDL, n_vectors, diagonal_frequency_dist)
        average_vertical_line_length = self.average_line_length(
            MVL, n_vectors, vertical_frequency_dist)
        average_white_vertical_line_length = self.average_line_length(
            MWVL, n_vectors, white_vertical_frequency_dist)

        longest_diagonal_line_length = self.longest_line_length(
            diagonal_frequency_dist, n_vectors, diag=True)
        longest_vertical_line_length = self.longest_line_length(
            vertical_frequency_dist, n_vectors, diag=False)
        longest_white_vertical_line_length = self.longest_line_length(
            white_vertical_frequency_dist, n_vectors, diag=False)

        entropy_diagonal_lines = self.entropy_lines(
            MDL, n_vectors, diagonal_frequency_dist, diag=True)
        entropy_vertical_lines = self.entropy_lines(
            MVL, n_vectors, vertical_frequency_dist, diag=False)
        entropy_white_vertical_lines = self.entropy_lines(
            MWVL, n_vectors, white_vertical_frequency_dist, diag=False)

        return {
            'RR': recurrence_rate,
            'DET': determinism,
            'ADLL': average_diagonal_line_length,
            'LDLL': longest_diagonal_line_length,
            'DIV': 1. / longest_diagonal_line_length,
            'EDL': entropy_diagonal_lines,
            'LAM': laminarity,
            'AVLL': average_vertical_line_length,
            'LVLL': longest_vertical_line_length,
            'EVL': entropy_vertical_lines,
            'AWLL': average_white_vertical_line_length,
            'LWLL': longest_white_vertical_line_length,
            'EWLL': entropy_white_vertical_lines,
            'RDRR': determinism / recurrence_rate,
            'RLD': laminarity / determinism}

    def calculate_vertical_frequency(self, number_of_vectors, not_white: int):
        vertical_frequency_distribution = np.zeros(number_of_vectors + 1)
        for i in range(number_of_vectors):
            vertical_line_length = 0
            for j in range(number_of_vectors):
                if self.recurrence_matrix[i, j] == not_white:
                    vertical_line_length += 1
                    if j == (number_of_vectors - 1):
                        vertical_frequency_distribution[vertical_line_length] += 1.0
                else:
                    if vertical_line_length != 0:
                        vertical_frequency_distribution[vertical_line_length] += 1.0
                        vertical_line_length = 0
        return vertical_frequency_distribution

    def calculate_diagonal_frequency(self, number_of_vectors):
        diagonal_frequency_distribution = np.zeros(number_of_vectors + 1)
        for i in range(number_of_vectors - 1, -1, -1):
            diagonal_line_length = 0
            for j in range(0, number_of_vectors - i):
                if self.recurrence_matrix[i + j, j] == 1:
                    diagonal_line_length += 1
                    if j == (number_of_vectors - i - 1):
                        diagonal_frequency_distribution[diagonal_line_length] += 1.0
                else:
                    if diagonal_line_length != 0:
                        diagonal_frequency_distribution[diagonal_line_length] += 1.0
                        diagonal_line_length = 0

        for k in range(1, number_of_vectors):
            diagonal_line_length = 0
            for i in range(number_of_vectors - k):
                j = i + k
                if self.recurrence_matrix[i, j] == 1:
                    diagonal_line_length += 1
                    if j == (number_of_vectors - 1):
                        diagonal_frequency_distribution[diagonal_line_length] += 1.0
                else:
                    if diagonal_line_length != 0:
                        diagonal_frequency_distribution[diagonal_line_length] += 1.0
                        diagonal_line_length = 0
        return diagonal_frequency_distribution

    def entropy_lines(
            self,
            factor,
            number_of_vectors,
            distribution,
            diag: bool):
        if diag:
            sum_frequency_distribution = float(np.sum(distribution[factor:-1]))
        else:
            number_of_vectors = number_of_vectors + 1
            sum_frequency_distribution = float(np.sum(distribution[factor:]))

        entropy_lines = 0
        for i in range(factor, number_of_vectors):
            if distribution[i] != 0:
                entropy_lines += (distribution[i] / sum_frequency_distribution) * \
                    np.log(distribution[i] / sum_frequency_distribution)
        return -entropy_lines

    def laminarity_or_determinism(
            self,
            factor,
            number_of_vectors,
            distribution,
            lam: bool):
        if lam:
            number_of_vectors = number_of_vectors + 1
        numerator = np.sum([i * distribution[i]
                            for i in range(factor, number_of_vectors)])
        denominator = np.sum([i * distribution[i]
                              for i in range(1, number_of_vectors)])
        return numerator / denominator

    def longest_line_length(
            self,
            frequency_distribution,
            number_of_vectors,
            diag: bool):
        longest_line_length = 1
        for i in range(number_of_vectors, 0, -1):
            if frequency_distribution[i] != 0:
                return i
        return longest_line_length

    def average_line_length(self, factor, number_of_vectors, distribution):
        numerator = np.sum([i * distribution[i]
                            for i in range(factor, number_of_vectors + 1)])
        denominator = np.sum([distribution[i]
                              for i in range(factor, number_of_vectors + 1)])
        return numerator / denominator
