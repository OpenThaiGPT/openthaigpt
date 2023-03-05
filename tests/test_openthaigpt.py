#!/usr/bin/env python

"""Tests for `openthaigpt` package."""


import unittest

from openthaigpt import openthaigpt_module

class TestOpenthaigpt(unittest.TestCase):
    """Tests for `openthaigpt` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_0000_usage(self):
        """Test something."""
        answer = openthaigpt_module.generate("Q: สวัสดีครับ")
        assert answer is not None
