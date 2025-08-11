import pytest

from autonomy.models.model import Model


class TestModel:
  def test_constructor_sets_name_correctly(self):
    expected = "test_model"
    model = Model(expected)
    assert model.name == expected
