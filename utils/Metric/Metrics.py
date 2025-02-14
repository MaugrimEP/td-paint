from conf.model_params import MetricsParams
from utils.Metric.MetricCelebA import CelebAMetrics, MockMetrics


def get_metrics(params: MetricsParams):
    if params.no_metrics:
        print("No metrics used, will use the MockMetrics")
        return MockMetrics

    metrics = {
        "celeba": CelebAMetrics,
    }

    if params.name not in metrics:
        raise ValueError(f"Unknown metric {params.name}")

    metric_constructor = metrics[params.name]
    
        

    return metric_constructor
