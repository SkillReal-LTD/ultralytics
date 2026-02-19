# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import os

from ultralytics.utils import LOGGER, RANK, SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["wandb"] is True  # verify integration is enabled
    import wandb as wb

    assert hasattr(wb, "__version__")  # verify package is not directory
    _processed_plots = {}
    _last_logged_epoch = -1  # Defense 1: prevent duplicate logs from final_eval
    _last_committed_step = 0  # Defense 2: safety net tracking actual wandb step

except (ImportError, AssertionError) as e:
    if RANK in {-1, 0}:
        if isinstance(e, ImportError):
            LOGGER.info(
                f"W&B: Not installed or import failed. Install with 'pip install wandb' to enable W&B logging. (RANK={RANK})"
            )
        elif isinstance(e, AssertionError):
            if TESTS_RUNNING:
                LOGGER.info(f"W&B: Disabled during testing (RANK={RANK})")
            elif SETTINGS.get("wandb") is not True:
                LOGGER.info(
                    f"W&B: Disabled in settings. Set 'wandb: True' in settings to enable. Current value: {SETTINGS.get('wandb', 'not set')} (RANK={RANK})"
                )
            else:
                LOGGER.info(f"W&B: Integration check failed (RANK={RANK})")
    wb = None


def _custom_table(x, y, classes, title="Precision Recall Curve", x_title="Recall", y_title="Precision"):
    """Create and log a custom metric visualization to wandb.plot.pr_curve.

    This function crafts a custom metric visualization that mimics the behavior of the default wandb precision-recall
    curve while allowing for enhanced customization. The visual metric is useful for monitoring model performance across
    different classes.

    Args:
        x (list): Values for the x-axis; expected to have length N.
        y (list): Corresponding values for the y-axis; also expected to have length N.
        classes (list): Labels identifying the class of each point; length N.
        title (str, optional): Title for the plot.
        x_title (str, optional): Label for the x-axis.
        y_title (str, optional): Label for the y-axis.

    Returns:
        (wandb.Object): A wandb object suitable for logging, showcasing the crafted metric visualization.
    """
    import polars as pl  # scope for faster 'import ultralytics'
    import polars.selectors as cs

    df = pl.DataFrame({"class": classes, "y": y, "x": x}).with_columns(cs.numeric().round(3))
    data = df.select(["class", "y", "x"]).rows()

    fields = {"x": "x", "y": "y", "class": "class"}
    string_fields = {"title": title, "x-axis-title": x_title, "y-axis-title": y_title}
    return wb.plot_table(
        "wandb/area-under-curve/v0",
        wb.Table(data=data, columns=["class", "y", "x"]),
        fields=fields,
        string_fields=string_fields,
    )


def _plot_curve(
    x,
    y,
    names=None,
    id="precision-recall",
    title="Precision Recall Curve",
    x_title="Recall",
    y_title="Precision",
    num_x=100,
    only_mean=False,
):
    """Log a metric curve visualization.

    This function generates a metric curve based on input data and logs the visualization to wandb. The curve can
    represent aggregated data (mean) or individual class data, depending on the 'only_mean' flag.

    Args:
        x (np.ndarray): Data points for the x-axis with length N.
        y (np.ndarray): Corresponding data points for the y-axis with shape (C, N), where C is the number of classes.
        names (list, optional): Names of the classes corresponding to the y-axis data; length C.
        id (str, optional): Unique identifier for the logged data in wandb.
        title (str, optional): Title for the visualization plot.
        x_title (str, optional): Label for the x-axis.
        y_title (str, optional): Label for the y-axis.
        num_x (int, optional): Number of interpolated data points for visualization.
        only_mean (bool, optional): Flag to indicate if only the mean curve should be plotted.

    Notes:
        The function leverages the '_custom_table' function to generate the actual visualization.
    """
    import numpy as np

    # Create new x
    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)

    # Create arrays for logging
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()

    if only_mean:
        table = wb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        wb.run.log({title: wb.plot.line(table, x_title, y_title, title=title)})
    else:
        classes = ["mean"] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)  # add new x
            y_log.extend(np.interp(x_new, x, yi))  # interpolate y to new x
            classes.extend([names[i]] * len(x_new))  # add class names
        wb.log({id: _custom_table(x_log, y_log, classes, title, x_title, y_title)}, commit=False)


def _log_plots(plots, step):
    """Log plots to WandB at a specific step if they haven't been logged already.

    This function checks each plot in the input dictionary against previously processed plots and logs new or updated
    plots to WandB at the specified step.

    Args:
        plots (dict): Dictionary of plots to log, where keys are plot names and values are dictionaries containing plot
            metadata including timestamps.
        step (int): The step/epoch at which to log the plots in the WandB run.

    Notes:
        The function uses a shallow copy of the plots dictionary to prevent modification during iteration.
        Plots are identified by their stem name (filename without extension).
        Each plot is logged as a WandB Image object.
    """
    for name, params in plots.copy().items():  # shallow copy to prevent plots dict changing during iteration
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step, commit=False)
            _processed_plots[name] = timestamp


def on_pretrain_routine_start(trainer):
    """Initialize and start wandb project if module is present."""
    if not wb.run:
        # Check for existing run ID from external WandbManager
        run_id = os.getenv("WANDB_RUN_ID")
        project = os.getenv("WANDB_PROJECT")

        if run_id and project:
            # Resume existing run created by external WandbManager
            LOGGER.info(f"W&B: Resuming run from external WandbManager (project={project}, id={run_id})")
            wb.init(
                project=project,
                id=run_id,
                resume="allow",  # Resume if exists, create if not
                config=vars(trainer.args),
            )
            LOGGER.info(f"W&B: Successfully resumed run at {wb.run.get_url()}")
        else:
            # Create new run (original behavior)
            LOGGER.info(
                f"W&B: Initializing new run (project={trainer.args.project or 'Ultralytics'}, name={trainer.args.name})"
            )
            wb.init(
                project=str(trainer.args.project).replace("/", "-") if trainer.args.project else "Ultralytics",
                name=str(trainer.args.name).replace("/", "-"),
                config=vars(trainer.args),
            )
            LOGGER.info(f"W&B: Successfully created new run at {wb.run.get_url()}")
    else:
        LOGGER.info("W&B: Run already initialized outside Ultralytics")


def on_fit_epoch_end(trainer):
    """Log training metrics and model information at the end of an epoch."""
    global _last_logged_epoch, _last_committed_step

    # Defense 1: skip duplicate call from final_eval()
    if trainer.epoch == _last_logged_epoch:
        return
    _last_logged_epoch = trainer.epoch

    step = trainer.epoch + 1

    # Defense 2: ensure monotonically increasing step
    if step <= _last_committed_step:
        step = _last_committed_step + 1

    _log_plots(trainer.plots, step=step)
    _log_plots(trainer.validator.plots, step=step)
    if trainer.epoch == 0:
        wb.run.log(model_info_for_loggers(trainer), step=step, commit=False)
    wb.run.log(trainer.metrics, step=step, commit=True)  # commit forces sync
    _last_committed_step = step


def on_train_epoch_end(trainer):
    """Log metrics and save images at the end of each training epoch."""
    step = trainer.epoch + 1
    if step <= _last_committed_step:
        return
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=step, commit=False)
    wb.run.log(trainer.lr, step=step, commit=False)
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=step)


def on_train_end(trainer):
    """Save the best model as an artifact and log final plots at the end of training."""
    # Use epoch + 2 to avoid step conflict with on_fit_epoch_end which commits at epoch + 1
    # After commit, wandb's internal step advances, so we need to log at a higher step
    final_step = max(trainer.epoch + 2, _last_committed_step + 1)
    _log_plots(trainer.validator.plots, step=final_step)
    _log_plots(trainer.plots, step=final_step)
    art = wb.Artifact(type="model", name=f"run_{wb.run.id}_model")
    if trainer.best.exists():
        art.add_file(trainer.best)
        wb.run.log_artifact(art, aliases=["best"])
    # Check if we actually have plots to save
    if trainer.args.plots and hasattr(trainer.validator.metrics, "curves_results"):
        for curve_name, curve_values in zip(trainer.validator.metrics.curves, trainer.validator.metrics.curves_results):
            x, y, x_title, y_title = curve_values
            _plot_curve(
                x,
                y,
                names=list(trainer.validator.metrics.names.values()),
                id=f"curves/{curve_name}",
                title=curve_name,
                x_title=x_title,
                y_title=y_title,
            )
    wb.run.finish()  # required or run continues on dashboard


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if wb
    else {}
)
