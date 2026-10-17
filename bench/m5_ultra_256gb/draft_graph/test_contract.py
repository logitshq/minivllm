"""CPU-only ownership and frozen-parameter contract, with compilation mocked."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from candidate import BackboneGraph


class Layer:
    def __init__(self):
        self.weight = object()
        self.ctx_k = object()
        self.ctx_v = object()

    def parameters(self):
        return {'weight': self.weight, 'self_attn': {'ctx_k': self.ctx_k, 'ctx_v': self.ctx_v}}


class FrozenContract(unittest.TestCase):
    def make_graph(self):
        layer = Layer()
        with patch('candidate.mx.compile', side_effect=lambda fn: fn):
            graph = BackboneGraph(SimpleNamespace(layers=[layer]))
        return graph, layer

    def test_request_cache_replacement_does_not_replace_weights(self):
        graph, layer = self.make_graph()
        layer.ctx_k, layer.ctx_v = object(), object()
        graph.assert_frozen()

    def test_parameter_replacement_requires_rebuild(self):
        graph, layer = self.make_graph()
        layer.weight = object()
        with self.assertRaisesRegex(RuntimeError, 'rebuild graph'):
            graph.assert_frozen()


if __name__ == '__main__':
    unittest.main()
