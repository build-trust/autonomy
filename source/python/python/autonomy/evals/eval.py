from dataclasses import dataclass, field
from typing import Protocol, Any, Callable, Dict, List, Union


class Metric(Protocol):
  @property
  def name(self) -> str: ...

  def calculate(self, expected_output: Any, observed_output: Any) -> float: ...

  def aggregate(self, values: List[float]) -> float: ...


@dataclass
class TestCase:
  input: Any
  expected_output: Any


@dataclass
class TestOk:
  test_case: TestCase
  output: Any
  metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestError:
  test_case: TestCase
  error: str


class Eval:
  def run(
    self, subject: Callable, cases: List[TestCase], metrics: List[Metric]
  ) -> tuple[Dict[str, float], List[Union[TestOk, TestError]]]:
    if not callable(subject):
      raise ValueError("subject is not callable")

    if not cases:
      raise ValueError("test case list is empty")

    if not metrics:
      raise ValueError("list of metrics is empty")

    results = []
    for case in cases:
      try:
        observed_output = subject(case.input)

        # Calculate all metrics for this test case
        calculated = {}
        for m in metrics:
          calculated[m.name] = m.calculate(case.expected_output, observed_output)

        results.append(TestOk(test_case=case, output=observed_output, metrics=calculated))
      except Exception as e:
        results.append(TestError(test_case=case, error=str(e)))

    # Aggregate each metric across all test cases
    aggregated = {}
    for m in metrics:
      values = []
      for result in results:
        if isinstance(result, TestOk) and m.name in result.metrics:
          values.append(result.metrics[m.name])
      aggregated[m.name] = m.aggregate(values) if values else float("nan")

    return aggregated, results
