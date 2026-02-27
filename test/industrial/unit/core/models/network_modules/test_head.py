import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

# Import necessary modules
from fedot_ind.core.models.nn.network_modules.layers.head_layers import (
    create_pool_head, max_pool_head, create_pool_plus_head, create_conv_head,
    create_mlp_head, create_fc_head, create_rnn_head, imputation_head,
    CreateConvLinNDHead, LinNDHead, RocketNDHead, Xresnet1dNDHead,
    CreateConv3dHead, universal_pool_head
)


class TestPoolHeads:
    def test_create_pool_head_basic(self):
        # Test basic pool head without options
        n_in, output_dim = 64, 10
        head = create_pool_head(n_in, output_dim)

        # Check layer composition
        assert isinstance(head, nn.Sequential)
        assert len(head) == 2
        assert isinstance(head[0], nn.Module)  # GAP1d
        assert isinstance(head[1], nn.Module)  # LinBnDrop

        # Test forward pass with dummy input
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)

    def test_create_pool_head_with_concat(self):
        # Test pool head with concat_pool=True
        n_in, output_dim = 64, 10
        head = create_pool_head(n_in, output_dim, concat_pool=True)

        # With concat_pool, n_in should be doubled
        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)

    def test_create_pool_head_with_y_range(self):
        # Test pool head with y_range
        n_in, output_dim = 64, 10
        y_range = (0, 1)
        head = create_pool_head(n_in, output_dim, y_range=y_range)

        # Check for SigmoidRange layer
        assert isinstance(head, nn.Sequential)
        assert len(head) == 3

        # Test forward pass with dummy input
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)
        # Check if output is within the specified range
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_max_pool_head(self):
        n_in, output_dim, seq_len = 64, 10, 20
        head = max_pool_head(n_in, output_dim, seq_len)

        assert isinstance(head, nn.Sequential)
        assert isinstance(head[0], nn.MaxPool1d)

        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)


class TestPoolPlusHead:
    def _test_create_pool_plus_head_basic(self):
        n_in, output_dim = 64, 10
        head = create_pool_plus_head(n_in, output_dim)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)

    def test_create_pool_plus_head_custom_layers(self):
        n_in, output_dim = 64, 10
        lin_ftrs = [128, 64]
        head = create_pool_plus_head(n_in, output_dim, lin_ftrs=lin_ftrs, fc_dropout=[0.5])

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)


class TestConvHead:
    def test_create_conv_head(self):
        n_in, output_dim = 64, 10
        head = create_conv_head(n_in, output_dim)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)

    def test_create_conv_head_with_adaptive_size(self):
        n_in, output_dim = 64, 10
        adaptive_size = 5
        head = create_conv_head(n_in, output_dim, adaptive_size=adaptive_size)

        assert isinstance(head, nn.Sequential)
        assert isinstance(head[0], nn.AdaptiveAvgPool1d)

        # Test forward pass
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)


class TestMlpHead:
    def test_create_mlp_head_with_flatten(self):
        n_in, output_dim, seq_len = 64, 10, 20
        head = create_mlp_head(n_in, output_dim, seq_len=seq_len, flatten=True)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)

    def _test_create_mlp_head_without_flatten(self):
        n_in, output_dim = 64, 10
        head = create_mlp_head(n_in, output_dim, flatten=False)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)


class TestFCHead:
    def test_create_fc_head_with_flatten(self):
        n_in, output_dim, seq_len = 64, 10, 20
        head = create_fc_head(n_in, output_dim, seq_len=seq_len, flatten=True)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)

    def _test_create_fc_head_custom_layers(self):
        n_in, output_dim = 64, 10
        lin_ftrs = [128, 64]
        head = create_fc_head(n_in, output_dim, lin_ftrs=lin_ftrs, fc_dropout=0.5, flatten=True,
                              seq_len=10)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)

    def _test_create_fc_head_with_y_range(self):
        n_in, output_dim = 64, 10
        y_range = (0, 1)
        head = create_fc_head(n_in, output_dim, y_range=y_range)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)
        # Check if output is within the specified range
        assert torch.all(output >= 0) and torch.all(output <= 1)


class TestRNNHead:
    def test_create_rnn_head(self):
        n_in, output_dim = 64, 10
        head = create_rnn_head(n_in, output_dim)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)


class TestImputationHead:
    def test_imputation_head(self):
        input_dim, output_dim = 64, 10
        head = imputation_head(input_dim, output_dim)

        assert isinstance(head, nn.Sequential)
        assert isinstance(head[1], nn.Conv1d)

        # Test forward pass
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, output_dim, seq_len)  # Note the dimension ordering
        output = head(x)

        assert output.shape == (batch_size, output_dim, seq_len)

    def test_imputation_head_with_y_range(self):
        input_dim, output_dim = 64, 10
        y_range = (0, 1)
        head = imputation_head(input_dim, output_dim, y_range=y_range)

        assert isinstance(head, nn.Sequential)
        assert len(head) == 3

        # Test forward pass
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, output_dim, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim, seq_len)
        # Check if output is within the specified range
        assert torch.all(output >= 0) and torch.all(output <= 1)


class TestNDHeads:
    def test_create_conv_lin_nd_head(self):
        n_in, n_out, seq_len, d = 64, 10, 20, 5
        head = CreateConvLinNDHead(n_in, n_out, seq_len, d)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, d, n_out)

    def test_create_conv_lin_nd_head_list_d(self):
        n_in, n_out, seq_len = 64, 10, 20
        d = [5, 4]
        head = CreateConvLinNDHead(n_in, n_out, seq_len, d)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, 5, 4, n_out)

    def test_lin_nd_head(self):
        n_in, n_out, seq_len, d = 64, 10, 20, 5
        head = LinNDHead(n_in, n_out, seq_len, d)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, d, n_out)

    def test_lin_nd_head_no_d(self):
        n_in, n_out, seq_len = 64, 10, 20
        head = LinNDHead(n_in, n_out, seq_len)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, n_out)

    def _test_rocket_nd_head(self):
        n_in, n_out, d = 64, 10, 5
        head = RocketNDHead(n_in, n_out, d=d)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, d, n_out)

    def test_xresnet1d_nd_head(self):
        n_in, n_out, d = 64, 10, 5
        head = Xresnet1dNDHead(n_in, n_out, d=d)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, d, n_out)

    def _test_create_conv3d_head(self):
        n_in, n_out, seq_len, d = 64, 10, 5, 5  # must be equal
        head = CreateConv3dHead(n_in, n_out, seq_len, d)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, n_out, d)

    def test_create_conv3d_head_with_mismatch_dims(self):
        n_in, n_out, seq_len, d = 64, 10, 20, 5

        # This should raise an assertion error since seq_len != d
        with pytest.raises(AssertionError):
            CreateConv3dHead(n_in, n_out, seq_len, d)


class TestUniversalPoolHead:
    def test_universal_pool_head(self):
        n_in, output_dim, seq_len = 64, 10, 20
        head = universal_pool_head(n_in, output_dim, seq_len)

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)

    def test_universal_pool_head_custom_params(self):
        n_in, output_dim, seq_len = 64, 10, 20
        head = universal_pool_head(
            n_in, output_dim, seq_len,
            mult=3, pool_n_layers=3,
            pool_dropout=0.3, fc_dropout=0.2
        )

        assert isinstance(head, nn.Sequential)

        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, n_in, seq_len)
        output = head(x)

        assert output.shape == (batch_size, output_dim)


@pytest.mark.parametrize("head_func", [
    create_pool_head,
    max_pool_head,
    create_pool_plus_head,
    create_conv_head,
    create_mlp_head,
    create_fc_head,
    create_rnn_head
])
def _test_head_func_with_kwargs(head_func):
    """Test that head functions handle unknown kwargs properly."""
    n_in, output_dim = 64, 10
    seq_len = 20 if head_func in [max_pool_head, create_mlp_head, create_fc_head] else None

    # Mock print to check if warning is printed
    with patch('builtins.print') as mock_print:
        if seq_len:
            head = head_func(n_in, output_dim, seq_len=seq_len,
                             #  unknown_kwarg=True
                             )
        else:
            head = head_func(n_in, output_dim,
                             #  unknown_kwarg=True
                             )

        # Check that print was called with a message about unused kwargs
        mock_print.assert_called_once()
        assert "not being used" in mock_print.call_args[0][0]

    assert isinstance(head, nn.Sequential)


def test_aliases():
    """Test that aliases point to the correct functions."""
    from fedot_ind.core.models.nn.network_modules.layers.head_layers import (
        pool_head, average_pool_head, concat_pool_head, pool_plus_head,
        conv_head, mlp_head, fc_head, rnn_head, conv_lin_nd_head,
        conv_lin_3d_head, create_conv_lin_3d_head, conv_3d_head,
        create_lin_nd_head, lin_3d_head, create_lin_3d_head
    )

    assert pool_head == create_pool_head
    assert average_pool_head.__name__ == "average_pool_head"
    assert concat_pool_head.__name__ == "concat_pool_head"
    assert pool_plus_head == create_pool_plus_head
    assert conv_head == create_conv_head
    assert mlp_head == create_mlp_head
    assert fc_head == create_fc_head
    assert rnn_head == create_rnn_head
    assert conv_lin_nd_head == CreateConvLinNDHead
    assert conv_lin_3d_head == CreateConvLinNDHead
    assert create_conv_lin_3d_head == CreateConvLinNDHead
    assert conv_3d_head == CreateConv3dHead
    assert create_lin_nd_head == LinNDHead
    assert lin_3d_head == LinNDHead
    assert create_lin_3d_head == LinNDHead


def test_heads_list():
    """Test that the heads list contains all expected head functions."""
    from fedot_ind.core.models.nn.network_modules.layers.head_layers import heads

    expected_heads = [
        create_mlp_head,
        create_fc_head,
        # average_pool_head,
        max_pool_head,
        # concat_pool_head,
        create_pool_plus_head,
        create_conv_head,
        create_rnn_head,
        CreateConvLinNDHead,
        LinNDHead,
        CreateConv3dHead
    ]

    # Check that all expected heads are in the heads list
    for head in expected_heads:
        assert head in heads
