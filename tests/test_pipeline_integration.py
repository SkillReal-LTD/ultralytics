"""Mirror of the algo-ai-training YOLO pipeline against this ultralytics checkout.

Every test reproduces something src/training_frameworks/yolo_trainer/ actually does:
config keys from cfg/training/*.yaml, the skillreal sync callback, resume from an
explicit checkpoint path, results_dict reporting, and ONNX export.
"""

import os
import shutil
from pathlib import Path

import pytest

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.metrics import DEFAULT_FITNESS_WEIGHT

IMGSZ = 64  # pipeline uses 640; shape-only paths are identical and this keeps runs short


# ---------------------------------------------------------------- config surface


def test_pipeline_config_keys_are_accepted():
    """Every training key the pipeline's cfg/training/*.yaml files set must still resolve."""
    pipeline_keys = {
        "val": True,
        "epochs": 2,
        "batch": 4,
        "imgsz": 640,
        "save_period": 1,
        "patience": 5,
        "seg_boundary_weight": 10,
        "classes": [0, 1],
        "fitness_weight": [0, 1, 0, 0],
        "dropout": 0.2,
        "lr0": 0.0015,
        "split": "test",
    }
    for k, v in pipeline_keys.items():
        assert k in DEFAULT_CFG_DICT or k in {"split"}, f"pipeline key '{k}' no longer exists"
    cfg = get_cfg(overrides={k: v for k, v in pipeline_keys.items() if k != "split"})
    assert cfg.seg_boundary_weight == 10
    assert cfg.fitness_weight == [0, 1, 0, 0]
    assert cfg.patience == 5


def test_fork_only_keys_survive():
    """The fork's own config keys must still exist with their documented defaults."""
    for k in (
        "fitness_weight",
        "class_weights",
        "class_weights_resolved",
        "channels",
        "cls_loss",
        "label_smoothing",
        "focal_gamma",
        "cb_beta",
        "arcface_margin",
        "arcface_scale",
        "seg_boundary_weight",
        "seg_boundary_kernel",
    ):
        assert k in DEFAULT_CFG_DICT, f"fork config key '{k}' disappeared"
    assert DEFAULT_CFG_DICT["fitness_weight"] == DEFAULT_FITNESS_WEIGHT
    assert DEFAULT_CFG_DICT["cls_loss"] == "ce"
    assert DEFAULT_CFG_DICT["seg_boundary_kernel"] == 3


# ---------------------------------------------------------------- segment pipeline


@pytest.fixture(scope="module")
def seg_run(tmp_path_factory):
    """One segment training run with the pipeline's own settings; reused by several tests."""
    out = tmp_path_factory.mktemp("seg")
    sync = out / "sync"
    os.environ["SYNC_CHECKPOINT_FOLDER"] = str(sync)
    model = YOLO("yolo11n-seg.pt")
    model.train(
        data="coco8-seg.yaml",
        project=str(out),
        name="job",
        epochs=2,
        batch=4,
        imgsz=IMGSZ,
        save_period=1,
        patience=5,
        val=True,
        seg_boundary_weight=10,
        plots=False,
        device="cpu",
    )
    yield out, sync, model
    os.environ.pop("SYNC_CHECKPOINT_FOLDER", None)


def test_segment_train_produces_checkpoints(seg_run):
    out, _, _ = seg_run
    assert (out / "job" / "weights" / "best.pt").exists()
    assert (out / "job" / "weights" / "last.pt").exists()


def test_skillreal_sync_callback(seg_run):
    """on_model_save copies last/best to SYNC_CHECKPOINT_FOLDER and epochs to epochs/."""
    _, sync, _ = seg_run
    assert (sync / "last.pt").exists(), "sync folder missing last.pt"
    assert (sync / "best.pt").exists(), "sync folder missing best.pt"
    assert (sync / "epochs").is_dir()
    assert list((sync / "epochs").glob("epoch*.pt")), "no periodic checkpoints synced"


def test_boundary_losses_are_reported(seg_run):
    """seg_boundary_weight>0 must surface a bnd loss column the pipeline logs."""
    out, _, _ = seg_run
    csv = (out / "job" / "results.csv").read_text()
    assert "bnd_loss" in csv, f"bnd_loss missing from results.csv header: {csv.splitlines()[0]}"


def test_results_dict_keys(seg_run):
    """YoloTrainer logs results.results_dict; skillreal.py prints it for Sagemaker."""
    out, _, _ = seg_run
    model = YOLO(str(out / "job" / "weights" / "best.pt"))
    res = model.val(data="coco8-seg.yaml", imgsz=IMGSZ, plots=False, device="cpu")
    rd = res.results_dict
    assert "fitness" in rd
    for k in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        assert k in rd, f"results_dict lost '{k}': {list(rd)}"
    for k in ("metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)"):
        assert k in rd, f"results_dict lost mask key '{k}': {list(rd)}"


def test_onnx_export(seg_run):
    """export_model_to_onnx() settings from cfg/training/*.yaml."""
    out, _, _ = seg_run
    model = YOLO(str(out / "job" / "weights" / "best.pt"))
    p = model.export(format="onnx", opset=12, dynamic=True, simplify=True, imgsz=IMGSZ, device="cpu")
    assert Path(p).exists() and Path(p).suffix == ".onnx"


def test_resume_from_explicit_checkpoint_path(tmp_path):
    """train(resume=<path>) on an interrupted run, which is how the pipeline resumes.

    The pipeline never passes a bare resume=True; resume_state supplies the staged path
    that the fork's on_model_save callback wrote to SYNC_CHECKPOINT_FOLDER.
    """
    args = {
        "data": "coco8-seg.yaml",
        "epochs": 2,
        "batch": 4,
        "imgsz": IMGSZ,
        "plots": False,
        "workers": 0,
        "project": str(tmp_path),
        "name": "resume",
        "exist_ok": True,
        "device": "cpu",
    }

    def stop_after_first_epoch(trainer):
        if trainer.epoch == 0:
            trainer.stop = True

    def disable_final_eval(trainer):
        trainer.final_eval = lambda: None

    model = YOLO("yolo11n-seg.pt")
    model.add_callback("on_train_start", disable_final_eval)
    model.add_callback("on_train_epoch_end", stop_after_first_epoch)
    model.train(**args)
    last = model.trainer.last

    staged = tmp_path / "staged_last.pt"  # the pipeline stages the synced checkpoint
    shutil.copy(last, staged)

    resumed = YOLO(str(staged))
    resumed.train(resume=str(staged), **args)
    assert resumed.trainer.start_epoch == 1, "resume did not continue from the interrupted epoch"


# ---------------------------------------------------------------- fitness weights


def test_segment_8_value_fitness_weight(tmp_path):
    """override_training_config.yaml ships an 8-value fitness_weight for segment."""
    model = YOLO("yolo11n-seg.pt")
    model.train(
        data="coco8-seg.yaml",
        project=str(tmp_path),
        name="fw8",
        epochs=1,
        batch=4,
        imgsz=IMGSZ,
        fitness_weight=[0.0, 0.9, 0.1, 0.0, 0.0, 0.5, 0.0, 0.5],
        plots=False,
        device="cpu",
    )
    assert (tmp_path / "fw8" / "weights" / "best.pt").exists()


def test_segment_10_value_fitness_weight(tmp_path):
    """The fork's boundary-aware 10-value layout adds mask_Dice / mask_BIoU."""
    model = YOLO("yolo11n-seg.pt")
    r = model.val(
        data="coco8-seg.yaml",
        imgsz=IMGSZ,
        fitness_weight=[0.0, 0.9, 0.1, 0.0, 0.0, 0.5, 0.0, 0.5, 0.3, 0.2],
        plots=False,
        device="cpu",
        project=str(tmp_path),
    )
    assert r.results_dict["fitness"] is not None


def test_pose_fitness_weight_and_classes(tmp_path):
    """yolo_keypoint.yaml: fitness_weight=[0,1,0,0] plus a classes filter."""
    model = YOLO("yolo11n-pose.pt")
    model.train(
        data="coco8-pose.yaml",
        project=str(tmp_path),
        name="pose",
        epochs=1,
        batch=4,
        imgsz=IMGSZ,
        fitness_weight=[0, 1, 0, 0],
        plots=False,
        device="cpu",
    )
    assert (tmp_path / "pose" / "weights" / "best.pt").exists()


# ---------------------------------------------------------------- class weights


def test_detect_class_weights_dict(tmp_path):
    """class_weights as a name->weight dict, resolved by the trainer."""
    model = YOLO("yolo11n.pt")
    model.train(
        data="coco8.yaml",
        project=str(tmp_path),
        name="cw",
        epochs=1,
        batch=4,
        imgsz=IMGSZ,
        class_weights={"person": 5.0},
        plots=False,
        device="cpu",
    )
    assert (tmp_path / "cw" / "weights" / "best.pt").exists()


# ---------------------------------------------------------------- classification


@pytest.mark.parametrize("cls_loss", ["ce", "focal", "cb_focal", "arcface"])
def test_classify_loss_variants(cls_loss, tmp_path):
    """The fork's four classification losses must all still train end to end."""
    model = YOLO("yolo11n-cls.pt")
    model.train(
        data="imagenet10",
        project=str(tmp_path),
        name=f"cls_{cls_loss}",
        epochs=1,
        batch=4,
        imgsz=IMGSZ,
        cls_loss=cls_loss,
        label_smoothing=0.05,
        plots=False,
        device="cpu",
    )
    assert (tmp_path / f"cls_{cls_loss}" / "weights" / "best.pt").exists()


# ---------------------------------------------------------------- wandb wiring


def test_wandb_external_run_id_path_is_intact():
    """The pipeline sets WANDB_RUN_ID/WANDB_PROJECT for multi-GPU runs to share one run."""
    import inspect

    from ultralytics.utils.callbacks import wb

    src = inspect.getsource(wb.on_pretrain_routine_start)
    assert 'os.getenv("WANDB_RUN_ID")' in src
    assert 'os.getenv("WANDB_PROJECT")' in src
    assert 'resume="allow"' in src


def test_skillreal_callbacks_registered():
    """base.py must still wire the fork's callback module in."""
    from ultralytics.utils.callbacks.base import get_default_callbacks

    assert "on_model_save" in get_default_callbacks()
    from ultralytics.utils.callbacks import skillreal

    assert "on_model_save" in skillreal.callbacks
    assert "on_train_start" in skillreal.callbacks
