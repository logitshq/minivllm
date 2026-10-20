"""CPU composition lifetime only; old fixture models never execute Metal."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R))
from bench.m5_ultra_256gb.gdn_graph_combined import test_integration_cpu as prior
from bench.m5_ultra_256gb.gdn_batched_graph import integration as current


class Lifecycle(prior.Tests):
    def setUp(self):
        self.binding=patch.object(prior,'m',current);self.binding.start();self.addCleanup(self.binding.stop)

    def test_frozen_sources_are_checked_before_dependency_execution(self):
        pins=current.source_pins()
        self.assertEqual(pins[str(current.GDN)],current.GDN_SHA)
        self.assertEqual(pins[str(current.ORIGINAL)],current.ORIGINAL_SHA)
        with patch.object(current,'GDN_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'batched GDN source'):current._dependencies()

    def test_original_composition_namespace_is_not_modified(self):
        self.assertIsNot(current.install.__globals__,current.original.install.__globals__)
        self.assertIsNot(current._installed,current.original._installed)
        self.assertIs(current.install.__globals__['_dependencies'],current._dependencies)
        self.assertEqual(current.original.GDN.name,'candidate.py')
        self.assertEqual(current.original.GDN.parent.name,'gdn_prework_unbound')


if __name__=='__main__':unittest.main()
