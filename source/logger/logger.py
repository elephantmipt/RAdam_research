from typing import Dict, List  # isort:skip
import logging
import os
import sys
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from tqdm import tqdm

from catalyst.dl import utils
from catalyst.dl.core import LoggerCallback, RunnerState
from catalyst.dl.utils.formatters import TxtMetricsFormatter
from catalyst.utils import format_metric
from catalyst.utils.tensorboard import SummaryWriter


class TensorboardLogger(LoggerCallback):
    """
    Logger callback, translates ``state.metrics`` to tensorboard
    """

    def __init__(
        self,
        metric_names: List[str] = None,
        log_on_batch_end: bool = True,
        log_on_epoch_end: bool = True,
    ):
        """
        Args:
            metric_names (List[str]): list of metric names to log,
                if none - logs everything
            log_on_batch_end (bool): logs per-batch metrics if set True
            log_on_epoch_end (bool): logs per-epoch metrics if set True
        """
        super().__init__()
        self.metrics_to_log = metric_names
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

        if not (self.log_on_batch_end or self.log_on_epoch_end):
            raise ValueError("You have to log something!")

        self.loggers = dict()

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, mode: str, suffix=""
    ):
        if self.metrics_to_log is None:
            metrics_to_log = sorted(list(metrics.keys()))
        else:
            metrics_to_log = self.metrics_to_log

        for name in metrics_to_log:
            if name in metrics:
                self.loggers[mode].add_scalar(
                    f"{name}{suffix}", metrics[name], step
                )

    def on_loader_start(self, state):
        """Prepare tensorboard writers for the current stage"""
        lm = state.loader_name
        if lm not in self.loggers:
            log_dir = os.path.join(state.logdir, f"{lm}_log")
            self.loggers[lm] = SummaryWriter(log_dir)


    def on_batch_end(self, state: RunnerState):
        """Translate batch metrics to tensorboard"""
        model = state.model
        n_iter = state.epoch
        for name, param in model.named_parameters():
            if 'bn' not in name:            
                logger.add_histogram(name, param.grad, n_iter)

        if self.log_on_batch_end:
            mode = state.loader_name
            metrics_ = state.metrics.batch_values
            self._log_metrics(
                metrics=metrics_, step=state.step, mode=mode, suffix="/batch"
            )

    def on_loader_end(self, state: RunnerState):
        """Translate epoch metrics to tensorboard"""
        if self.log_on_epoch_end:
            mode = state.loader_name
            metrics_ = state.metrics.epoch_values[mode]
            self._log_metrics(
                metrics=metrics_,
                step=state.epoch_log,
                mode=mode,
                suffix="/epoch",
            )
        for logger in self.loggers.values():
            logger.flush()

    def on_stage_end(self, state: RunnerState):
        """Close opened tensorboard writers"""
        for logger in self.loggers.values():
            logger.close()
