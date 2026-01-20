from datetime import datetime
from pathlib import Path

import ignite
import ignite.distributed as idist
import torch
from ignite.contrib.engines import common
from ignite.engine import Engine, Events
from ignite.utils import manual_seed, setup_logger
from torch.cuda.amp import autocast

from utils.array_operations import to

from ignite.contrib.handlers.tensorboard_logger import *


def base_evaluation(
    local_rank,
    config,
    get_dataflow,
    initialize,
    get_metrics,
    logger=None,
    visualize=None,
):
    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)
    device = idist.device()

    ## To distinguish between original BTS model vs DFT model writing in tensorboard logger
    # if config["data"]["type"] == ("KITTI_Raw_DFT" or "KITTI_360_DFT"):
    #     model_conf = config["model_conf"]
    #     enc = model_conf["encoder"]
    #     dec_h = model_conf["decoder_heads"][0]
    #     dec_args = dec_h["args"]
    #     dec_emb = dec_args["embedding_encoder"]
    #     attn_layers = dec_args["attn_layers"]
    #     readout_token = attn_layers["readout_token"]
    #     data_fisheye = config["data"]["data_fisheye"]
    #     data_stereo = config["data"]["data_stereo"]
    #     frame_count = config["data"]["data_fc"]
    #     frame_sample_mode = model_conf["frame_sample_mode"]

    #     lr_ = config["learning_rate"]
    #     bs_ = config["batch_size"]

    #     rbs = model_conf["ray_batch_size"]
    #     z_mode = model_conf["code_mode"]

    #     frz = enc["freeze"]
    #     do_ = dec_args["dropout_views_rate"]
    #     do_h = dec_args["dropout_multiviewhead"]

    #     dec_type = dec_emb["type"]  ## ff
    #     dec_dout = dec_emb["d_out"]
    #     dec_IBR = attn_layers["IBRAttn"]
    #     dec_nly = attn_layers["n_layers"]
    #     dec_nh = attn_layers["n_heads"]

    #     readout_token_type = readout_token["type"]

    #     model_name = "Smode" + frame_sample_mode + "_Fe" + str(data_fisheye)[:1] + "_St" + str(data_stereo)[:1] + "Fr" + str(frz)[:1] \
    #     + "_Fc" + str(frame_count) + "_do" + str(do_) + "_doh" + str(do_h)[:1] + "_embEnc" + str(dec_type) + "_dout" + str(dec_dout) \
    #     + "_decIBR" + str(dec_IBR)[:1] + "_nly" + str(dec_nly) + "_nh" + str(dec_nh) + "_readoutType" + readout_token_type \
    #     + "_lr" + str(lr_) + "_bs" + str(bs_) + "_rbs" + str(rbs) + "_ztype_" + z_mode + "_trainType_" + config["name"]
    #     logger = setup_logger(model_name)
    # else:
    model_name = config["name"]
    logger = setup_logger(name=model_name)  ## default

    log_basic_info(logger, config)

    output_path = config["output_path"]
    if rank == 0:
        if config["stop_iteration"] is None:
            now = datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            now = f"stop-on-{config['stop_iteration']}"

        # folder_name = f"{config['name']}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
        folder_name = (
            f"{model_name}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
        )

        output_path = Path(output_path) / folder_name
        if not output_path.exists():
            output_path.mkdir(parents=True)
        config["output_path"] = output_path.as_posix()
        logger.info(f"Output path: {config['output_path']}")

        if "cuda" in device.type:
            config["cuda device name"] = torch.cuda.get_device_name(local_rank)

    # Setup dataflow, model, optimizer, criterion
    test_loader = get_dataflow(config)  ## default
    # test_loader = get_dataflow(config, logger)

    if hasattr(test_loader, "dataset"):
        logger.info(f"Dataset length: Test: {len(test_loader.dataset)}")

    config["num_iters_per_epoch"] = len(test_loader)
    model = initialize(config, logger)

    cp_path = config.get("checkpoint", None)

    if cp_path is not None:
        if not cp_path.endswith(".pt"):
            cp_path = Path(cp_path)
            cp_path = next(cp_path.glob("training*.pt"))
        checkpoint = torch.load(cp_path, map_location=device)
        print(f"__check point loaded in path: {cp_path}")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print("__Be careful, no model is loaded")
    model.to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Let's now setup evaluator engine to perform model's validation and compute metrics
    metrics = get_metrics(config, device)

    # We define two evaluators as they wont have exactly similar roles:
    # - `evaluator` will save the best model based on validation score
    evaluator = create_evaluator(model, metrics=metrics, config=config, logger=logger)

    # # Create Tensorboard logger
    # tb_logger = TensorboardLogger(log_dir="/tmp/tb_logs")

    evaluator.add_event_handler(
        Events.ITERATION_COMPLETED(every=config["log_every"]),
        log_metrics_current(logger, metrics),
    )
    # evaluator.add_event_handler(Events.ITERATION_COMPLETED(every=config["log_every"]), log_metrics_current(logger, metrics), tensorboard_metrics_logging(tb_logger, metrics, config["log_every"]))

    try:
        state = evaluator.run(test_loader, max_epochs=1)
        log_metrics(logger, state.times["COMPLETED"], "Test", state.metrics)
        logger.info(f"Checkpoint: {str(cp_path)}")
    except Exception as e:
        logger.exception("")
        raise e


def log_basic_info(logger, config):
    logger.info(f"Run {config['name']}")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(
            f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}"
        )
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


# # Attach the logger to your evaluator
# def tensorboard_metrics_logging(engine, logger, metrics, log_interval):
#     if engine.state.iteration % log_interval == 0:
#         for name, value in metrics.items():
#             logger.writer.add_scalar(name, value.compute(), engine.state.iteration)


def log_metrics_current(logger, metrics):
    def f(engine):
        out_str = "\n" + "\t".join(
            [
                f"{v.compute():.3f}".ljust(8)
                for k, v in metrics.items()
                if v._num_examples != 0
                and (k not in ["abs_errH", "rel_errH", "thresholdH"])
            ]
        )
        out_str += "\n" + "\t".join([f"{k}".ljust(8) for k in metrics.keys()])
        logger.info(out_str)

    return f


def log_metrics(logger, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEvaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


# def create_evaluator(model, metrics, config, tag="val"):
def create_evaluator(model, metrics, config, logger=None, tag="val"):
    with_amp = config["with_amp"]
    device = idist.device()

    @torch.no_grad()
    def evaluate_step(engine: Engine, data):
        # if not engine.state_dict["iteration"] % 10 == 0:      ## to prevent iterating whole testset for viz purpose
        model.eval()
        if "t__get_item__" in data:
            timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
        else:
            timing = {}

        data = to(data, device)

        with autocast(enabled=with_amp):
            data = model(data)  ## ! This is where the occupancy prediction is made.

        loss_metrics = {}

        return {
            "output": data,
            "loss_dict": loss_metrics,
            "timings_dict": timing,
            "metrics_dict": {},
        }

    evaluator = Engine(evaluate_step)
    evaluator.logger = logger  ##

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    if idist.get_rank() == 0 and (not config.get("with_clearml", False)):
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)

    return evaluator
