from __future__ import division, print_function

import torch


class RecurrenceFeatureExtractorTorch:
    """
    Extractor for recurrence plot features.

    This class computes various statistical features from a recurrence matrix,
    such as recurrence rate, determinism, laminarity, and line-based metrics.

    Attributes:
        recurrence_matrix (torch.Tensor): The input recurrence matrix as a PyTorch tensor.
    """

    def __init__(self, recurrence_matrix: torch.Tensor = None):
        self.recurrence_matrix = recurrence_matrix.float()

    def quantification_analysis(self, MDL: int = 3, MVL: int = 3, MWVL: int = 2):
        """
        Perform quantification analysis on the recurrence matrix.

        Args:
            MDL (int, optional): Minimum diagonal line length. Defaults to 3.
            MVL (int, optional): Minimum vertical line length. Defaults to 3.
            MWVL (int, optional): Minimum white vertical line length. Defaults to 2.

        Returns:
            dict: A dictionary containing the following RQA features:
                - 'RR': Recurrence rate.
                - 'DET': Determinism.
                - 'ADLL': Average diagonal line length.
                - 'LDLL': Longest diagonal line length.
                - 'DIV': Divergence (inverse of longest diagonal line length).
                - 'EDL': Entropy of diagonal lines.
                - 'LAM': Laminarity.
                - 'AVLL': Average vertical line length.
                - 'LVLL': Longest vertical line length.
                - 'EVL': Entropy of vertical lines.
                - 'AWLL': Average white vertical line length.
                - 'LWLL': Longest white vertical line length.
                - 'EWLL': Entropy of white vertical lines.
                - 'RDRR': Ratio of determinism to recurrence rate.
                - 'RLD': Ratio of laminarity to determinism.
        """

        n_vectors = self.recurrence_matrix.shape[0]
        recurrence_rate = torch.sum(self.recurrence_matrix) / (n_vectors ** 2)

        diagonal_frequency_dist = self.calculate_diagonal_frequency(n_vectors)
        vertical_frequency_dist = self.calculate_vertical_frequency(n_vectors, not_white=1)
        white_vertical_frequency_dist = self.calculate_vertical_frequency(n_vectors, not_white=0)

        determinism = self.laminarity_or_determinism(MDL, n_vectors, diagonal_frequency_dist, lam=False)
        laminarity = self.laminarity_or_determinism(MVL, n_vectors, vertical_frequency_dist, lam=True)

        average_diagonal_line_length = self.average_line_length(MDL, n_vectors, diagonal_frequency_dist)
        average_vertical_line_length = self.average_line_length(MVL, n_vectors, vertical_frequency_dist)
        average_white_vertical_line_length = self.average_line_length(MWVL, n_vectors, white_vertical_frequency_dist)

        longest_diagonal_line_length = self.longest_line_length(diagonal_frequency_dist, n_vectors)
        longest_vertical_line_length = self.longest_line_length(vertical_frequency_dist, n_vectors)
        longest_white_vertical_line_length = self.longest_line_length(white_vertical_frequency_dist, n_vectors)

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
            'DIV': (1. / longest_diagonal_line_length) if longest_diagonal_line_length > 0 else 0.,
            'EDL': entropy_diagonal_lines,
            'LAM': laminarity,
            'AVLL': average_vertical_line_length,
            'LVLL': longest_vertical_line_length,
            'EVL': entropy_vertical_lines,
            'AWLL': average_white_vertical_line_length,
            'LWLL': longest_white_vertical_line_length,
            'EWLL': entropy_white_vertical_lines,
            'RDRR': (determinism / recurrence_rate) if recurrence_rate > 0 else 0.,
            'RLD': (laminarity / determinism) if determinism > 0 else 0.,
        }

    def calculate_vertical_frequency(self, number_of_vectors, not_white: int):
        """
        Calculate the frequency distribution of vertical lines in the recurrence matrix.

        Args:
            number_of_vectors (int): The number of vectors (size of the recurrence matrix).
            not_white (int): Flag to distinguish between white (0) and non-white (1) vertical lines.

        Returns:
            torch.Tensor: Frequency distribution of vertical lines of different lengths.
                The i-th element represents the count of vertical lines of length i.
        """
        m = (self.recurrence_matrix == not_white).float()
        freq = torch.zeros(number_of_vectors + 1, device=m.device)
        pad0 = torch.zeros(1, device=m.device)

        for col in m:
            diff = torch.cat([pad0, col, pad0])
            edges = diff[1:] - diff[:-1]
            starts = torch.where(edges == 1)[0]
            ends = torch.where(edges == -1)[0]
            lengths = ends - starts
            freq.index_add_(0, lengths,
                            torch.ones(lengths.numel(),
                                       device=m.device, dtype=freq.dtype))
        return freq

    def calculate_diagonal_frequency(self, number_of_vectors):
        """
        Calculate the frequency distribution of diagonal lines in the recurrence matrix.

        Args:
            number_of_vectors (int): The number of vectors (size of the recurrence matrix).

        Returns:
            torch.Tensor: Frequency distribution of diagonal lines of different lengths.
                The i-th element represents the count of diagonal lines of length i.
        """

        m = self.recurrence_matrix.float()
        freq = torch.zeros(number_of_vectors + 1, device=m.device)
        pad0 = torch.zeros(1, device=m.device)

        for off in range(-(number_of_vectors - 1), number_of_vectors):
            diag = torch.diagonal(m, offset=off)
            if diag.numel() < 2:
                continue
            diff = torch.cat([pad0, diag, pad0])
            edges = diff[1:] - diff[:-1]
            starts = torch.where(edges == 1)[0]
            ends = torch.where(edges == -1)[0]
            lengths = ends - starts
            freq.index_add_(0, lengths, torch.ones(lengths.numel(),
                                                   device=freq.device))
        return freq

    def entropy_lines(self, factor: int, number_of_vectors: int,
                      distribution: torch.Tensor, diag: bool):
        """
        Calculate the entropy of diagonal or vertical lines in the recurrence matrix.

        Args:
            factor (int): Minimum line length to consider.
            number_of_vectors (int): The number of vectors (size of the recurrence matrix).
            distribution (torch.Tensor): Frequency distribution of line lengths.
            diag (bool): If True, calculate entropy for diagonal lines; otherwise, for vertical lines.

        Returns:
            float: The entropy of the line length distribution.
        """

        if diag:
            sum_freq = torch.sum(distribution[factor:-1])
            end = number_of_vectors - 1
        else:
            end = number_of_vectors + 1
            sum_freq = torch.sum(distribution[factor:end])
        if sum_freq == 0:
            return 0.0
        segment = distribution[factor:end]
        probs = segment / sum_freq
        mask = segment != 0
        probs = probs[mask]
        return -torch.sum(probs * torch.log(probs))

    def laminarity_or_determinism(self, factor, number_of_vectors, distribution, lam: bool):
        """
        Calculate laminarity or determinism from the line length distribution.

        Args:
            factor (int): Minimum line length to consider.
            number_of_vectors (int): The number of vectors (size of the recurrence matrix).
            distribution (torch.Tensor): Frequency distribution of line lengths.
            lam (bool): If True, calculate laminarity; otherwise, calculate determinism.

        Returns:
            float: The laminarity or determinism value.
        """
        if lam:
            number_of_vectors = number_of_vectors + 1
        idx = torch.arange(1, number_of_vectors, device=distribution.device, dtype=distribution.dtype)
        numerator = torch.sum(idx[factor - 1:] * distribution[factor:number_of_vectors])
        denominator = torch.sum(idx * distribution[1:number_of_vectors])
        return numerator / denominator

    def longest_line_length(self, frequency_distribution: torch.Tensor, number_of_vectors):
        """
        Find the length of the longest line in the frequency distribution.

        Args:
            frequency_distribution (torch.Tensor): Frequency distribution of line lengths.
            number_of_vectors (int): The number of vectors (size of the recurrence matrix).

        Returns:
            float: The length of the longest line. Returns 1.0 if no lines are found.
        """
        lines = torch.where(frequency_distribution > 0)[0]
        if len(lines) != 0:
            return lines[-1]
        return 1.

    def average_line_length(self, factor, number_of_vectors, distribution: torch.Tensor):
        """
        Calculate the average length of lines in the frequency distribution.

        Args:
            factor (int): Minimum line length to consider.
            number_of_vectors (int): The number of vectors (size of the recurrence matrix).
            distribution (torch.Tensor): Frequency distribution of line lengths.

        Returns:
            float: The average length of lines. Returns 0.0 if no lines are found.
        """
        i = torch.arange(number_of_vectors + 1, device=distribution.device)
        num = torch.sum(i[factor:] * distribution[factor:])
        den = torch.sum(distribution[factor:])
        return num / den if den > 0 else 0.
