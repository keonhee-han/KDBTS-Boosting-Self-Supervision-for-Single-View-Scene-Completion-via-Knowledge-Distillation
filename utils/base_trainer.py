import json
import time
from datetime import datetime
from pathlib import Path
from typing import Union

from omegaconf import OmegaConf

import ignite
import ignite.distributed as idist
import torch
from ignite.contrib.engines import common
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.base_logger import BaseHandler
from ignite.engine import Engine, Events, EventEnum
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.utils import manual_seed, setup_logger
from torch.cuda.amp import autocast, GradScaler

from utils.array_operations import to
from utils.metrics import MeanMetric


def base_training(local_rank, config, get_dataflow, initialize, get_metrics, visualize):
    rank = (
        idist.get_rank()
    )  ## rank of the current process within a group of processes: each process could handle a unique subset of the data, based on its rank
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

    #     model_name = (
    #         "Smode"
    #         + frame_sample_mode
    #         + "_Fe"
    #         + str(data_fisheye)[:1]
    #         + "_St"
    #         + str(data_stereo)[:1]
    #         + "Fr"
    #         + str(frz)[:1]
    #         + "_Fc"
    #         + str(frame_count)
    #         + "_do"
    #         + str(do_)
    #         + "_doh"
    #         + str(do_h)[:1]
    #         + "_embEnc"
    #         + str(dec_type)
    #         + "_dout"
    #         + str(dec_dout)
    #         + "_decIBR"
    #         + str(dec_IBR)[:1]
    #         + "_nly"
    #         + str(dec_nly)
    #         + "_nh"
    #         + str(dec_nh)
    #         + "_readoutType"
    #         + readout_token_type
    #         + "_lr"
    #         + str(lr_)
    #         + "_bs"
    #         + str(bs_)
    #         + "_rbs"
    #         + str(rbs)
    #         + "_ztype_"
    #         + z_mode
    #         + "_trainType_"
    #         + config["name"]
    #     )
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
        folder_name = f"{model_name}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"

        output_path = Path(output_path) / folder_name
        if not output_path.exists():
            output_path.mkdir(parents=True)
        config["output_path"] = output_path.as_posix()
        logger.info(f"Output path: {config['output_path']}")

        if "cuda" in device.type:
            config["cuda device name"] = torch.cuda.get_device_name(local_rank)

    # Setup dataflow, model, optimizer, criterion
    loaders = get_dataflow(config, logger)
    if len(loaders) == 2:
        train_loader, test_loader = loaders
        vis_loader = None
    else:
        train_loader, test_loader, vis_loader = loaders

    if hasattr(train_loader, "dataset"):
        logger.info(f"Dataset length: Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    config["num_iters_per_epoch"] = len(train_loader)
    model, optimizer, criterion, lr_scheduler = initialize(config, logger)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Let's now setup evaluator engine to perform model's validation and compute metrics
    metrics = get_metrics(config, device)
    metrics_loss = {
        k: MeanMetric((lambda y: lambda x: x["loss_dict"][y])(k)) for k in criterion.get_loss_metric_names()
    }

    loss_during_validation = config.get("loss_during_validation", True)
    if loss_during_validation:
        eval_metrics = {**metrics, **metrics_loss}
    else:
        eval_metrics = metrics

    # Create trainer for current task
    trainer = create_trainer(
        model,
        optimizer,
        criterion,
        lr_scheduler,
        train_loader.sampler if hasattr(train_loader, "sampler") else None,
        config,
        logger,
        metrics={},
    )

    # We define two evaluators as they wont have exactly similar roles:
    # - `evaluator` will save the best model based on validation score
    evaluator = create_evaluator(
        model, metrics=eval_metrics, criterion=criterion if loss_during_validation else None, config=config
    )

    if vis_loader is not None:
        visualizer = create_evaluator(
            model, metrics=eval_metrics, criterion=criterion if loss_during_validation else None, config=config
        )
    else:
        visualizer = None

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = evaluator.run(test_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Test", state.metrics)

    def run_visualization(engine):
        epoch = trainer.state.epoch
        state = visualizer.run(vis_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Vis", state.metrics)

    eval_use_iters = config.get("eval_use_iters", False)
    vis_use_iters = config.get("vis_use_iters", False)

    if not eval_use_iters:
        trainer.add_event_handler(
            # Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.COMPLETED,
            Events.EPOCH_COMPLETED(every=config["validate_every"]),
            run_validation,
        )
    else:
        trainer.add_event_handler(
            # Events.ITERATION_COMPLETED(every=config["validate_every"]) | Events.COMPLETED,
            Events.ITERATION_COMPLETED(every=config["validate_every"]),
            run_validation,
        )

    if visualizer:
        if not vis_use_iters:
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED(every=config["visualize_every"]) | Events.COMPLETED, run_visualization
            )
        else:
            trainer.add_event_handler(
                Events.ITERATION_COMPLETED(every=config["visualize_every"]) | Events.COMPLETED, run_visualization
            )

    if rank == 0:
        # Setup TensorBoard logging on trainer and evaluators. Logged values are:
        #  - Training metrics, e.g. running average loss values
        #  - Learning rate
        #  - Evaluation train/test metrics

        trainer_timer = IterationTimeHandler()
        trainer_timer_data = DataloaderTimeHandler()
        trainer.add_event_handler(Events.ITERATION_STARTED, trainer_timer.start_iteration)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, trainer_timer.end_iteration)
        trainer.add_event_handler(Events.GET_BATCH_STARTED, trainer_timer_data.start_get_batch)
        trainer.add_event_handler(Events.GET_BATCH_COMPLETED, trainer_timer_data.end_get_batch)

        evaluator_timer = IterationTimeHandler()
        evaluator_timer_data = DataloaderTimeHandler()
        evaluator.add_event_handler(Events.ITERATION_STARTED, evaluator_timer.start_iteration)
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, evaluator_timer.end_iteration)
        evaluator.add_event_handler(Events.GET_BATCH_STARTED, evaluator_timer_data.start_get_batch)
        evaluator.add_event_handler(Events.GET_BATCH_COMPLETED, evaluator_timer_data.end_get_batch)

        if visualizer:
            visualizer_timer = IterationTimeHandler()
            visualizer_timer_data = DataloaderTimeHandler()
            visualizer.add_event_handler(Events.ITERATION_STARTED, visualizer_timer.start_iteration)
            visualizer.add_event_handler(Events.ITERATION_COMPLETED, visualizer_timer.end_iteration)
            visualizer.add_event_handler(Events.GET_BATCH_STARTED, visualizer_timer_data.start_get_batch)
            visualizer.add_event_handler(Events.GET_BATCH_COMPLETED, visualizer_timer_data.end_get_batch)

        gst = lambda engine, event_name: trainer.state.epoch
        gst_it_epoch = (
            lambda engine, event_name: (trainer.state.epoch - 1) * engine.state.epoch_length
            + engine.state.iteration
            - 1
        )
        eval_gst_it_iters = (
            lambda engine, event_name: (
                ((trainer.state.epoch - 1) * trainer.state.epoch_length + trainer.state.iteration)
                // config["validate_every"]
            )
            * engine.state.epoch_length
            + engine.state.iteration
            - 1
        )
        vis_gst_it_iters = (
            lambda engine, event_name: (
                ((trainer.state.epoch - 1) * trainer.state.epoch_length + trainer.state.iteration)
                // config["visualize_every"]
            )
            * engine.state.epoch_length
            + engine.state.iteration
            - 1
        )

        eval_gst_ep_iters = lambda engine, event_name: (
            ((trainer.state.epoch - 1) * trainer.state.epoch_length + trainer.state.iteration)
            // config["validate_every"]
        )
        vis_gst_ep_iters = lambda engine, event_name: (
            ((trainer.state.epoch - 1) * trainer.state.epoch_length + trainer.state.iteration)
            // config["visualize_every"]
        )

        eval_gst_it = eval_gst_it_iters if eval_use_iters else gst_it_epoch
        vis_gst_it = vis_gst_it_iters if vis_use_iters else gst_it_epoch

        eval_gst_ep = eval_gst_ep_iters if eval_use_iters else gst
        vis_gst_ep = vis_gst_ep_iters if vis_use_iters else gst

        tb_logger = TensorboardLogger(log_dir=output_path)
        tb_logger.attach(
            trainer,
            MetricLoggingHandler("train", optimizer),
            Events.ITERATION_COMPLETED(every=config.get("log_every_iters", 1)),
        )
        tb_logger.attach(
            evaluator,
            MetricLoggingHandler("val", log_loss=False, global_step_transform=eval_gst_ep),
            Events.EPOCH_COMPLETED,
        )
        if visualizer:
            tb_logger.attach(
                visualizer,
                MetricLoggingHandler("vis", log_loss=False, global_step_transform=vis_gst_ep),
                Events.EPOCH_COMPLETED,
            )

        # Plot config to tensorboard
        config_json = json.dumps(OmegaConf.to_container(config, resolve=True), indent=2)
        config_json = "".join("\t" + line for line in config_json.splitlines(True))
        tb_logger.writer.add_text("config", text_string=config_json, global_step=0)

        if visualize is not None:
            train_log_interval = config.get("log_tb_train_every_iters", -1)
            val_log_interval = config.get("log_tb_val_every_iters", train_log_interval)
            vis_log_interval = config.get("log_tb_vis_every_iters", 1)

            if train_log_interval > 0:
                tb_logger.attach(
                    trainer,
                    VisualizationHandler(tag="training", visualizer=visualize),
                    Events.ITERATION_COMPLETED(every=train_log_interval),
                )
            if val_log_interval > 0:
                tb_logger.attach(
                    evaluator,
                    VisualizationHandler(tag="val", visualizer=visualize, global_step_transform=eval_gst_it),
                    Events.ITERATION_COMPLETED(every=val_log_interval),
                )
            if visualizer and vis_log_interval > 0:
                tb_logger.attach(
                    visualizer,
                    VisualizationHandler(tag="vis", visualizer=visualize, global_step_transform=vis_gst_it),
                    Events.ITERATION_COMPLETED(every=vis_log_interval) | Events.ITERATION_COMPLETED(every=vis_log_interval // 10, before=vis_log_interval),
                )

    if "save_best" in config:
        # Store 2 best models by validation accuracy starting from num_epochs / 2:
        save_best_config = config["save_best"]
        metric_name = save_best_config["metric"]
        sign = save_best_config.get("sign", 1.0)

        best_model_handler = Checkpoint(
            {"model": model},
            get_save_handler(config),
            filename_prefix="best",
            n_saved=5,
            global_step_transform=global_step_from_engine(trainer),
            score_name=metric_name,
            score_function=Checkpoint.get_default_score_fn(metric_name, score_sign=sign),
        )
        evaluator.add_event_handler(
            Events.COMPLETED(lambda *_: trainer.state.epoch > config["num_epochs"] // 2), best_model_handler
        )

    # In order to check training resuming we can stop training on a given iteration
    if config["stop_iteration"] is not None:

        @trainer.on(Events.ITERATION_STARTED(once=config["stop_iteration"]))
        def _():
            logger.info(f"Stop training on {trainer.state.iteration} iteration")
            trainer.terminate()

    try:  ## train_loader == models.bts.trainer_overfit.DataloaderDummy object
        trainer.run(train_loader, max_epochs=config["num_epochs"])
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        tb_logger.close()


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}")


def log_basic_info(logger, config):
    logger.info(f"Run {config['name']}")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
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


def create_trainer(model, optimizer, criterion, lr_scheduler, train_sampler, config, logger, metrics={}):
    device = idist.device()

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    # hook_fwd_handles, hook_bwd_handles = []
    # for handle in hook_fw_handles:  handle.remove()   ## remove hooks for safety
    with_amp = config["with_amp"]
    requires_pgt = config["loss"]["lambda_pseudo_ground_truth"]

    scaler = GradScaler(enabled=with_amp)

    if model.renderer.net.__class__.__name__ == "MVBTSNet" and (requires_pgt != 0):
        invalid_features = []
        head_outputs = {name: [] for name, _ in model.renderer.net.heads.items()}

        def hook_fn_forward_invalid():
            def _hook_fn(module, input, output):
                if module.training:
                    invalid_features.append(module.invalid_features.flatten(0, 1))

            return _hook_fn

        def hook_fn_forward_heads(name):
            def _hook_fn(module, input, output):
                head_outputs[name].append(output)
                # head_outputs[name] = output

            return _hook_fn

    def train_step(engine, data: dict):
        if "t__get_item__" in data:
            timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
        else:
            timing = {}

        _start_time = time.time()

        # if model.renderer.net.__class__.__name__ == "MVBTSNet":
        #     invalid_features = []
        #     head_outputs = {name: [] for name, _ in model.renderer.net.heads.items()}

        data = to(data, device)

        if (model.renderer.net.__class__.__name__ == "MVBTSNet") and (requires_pgt != 0):
            nonlocal head_outputs
            nonlocal invalid_features
            #     head_outputs = {name: [] for name, _ in model.renderer.net.heads.items()}
            #     head_outputs = model.renderer.net.mlp_coarse

            if not model.renderer.net._forward_hooks:
                print(f"__Registering invalid_feature_hook fwd hook")
                model.renderer.net.register_forward_hook(hook_fn_forward_invalid())

            for name, module in model.renderer.net.heads.items():
                if not module._forward_hooks:
                    print(f"__Registering {name} fwd hook")
                    module.register_forward_hook(hook_fn_forward_heads(name))
                # if not module._backward_hooks:
                #     print(f"__Registering {name} bwd hook")
                #     module.register_backward_hook(lambda module, gin, gout: print(gout[0].mean()))
                # else: print(f"__Not registering for {name}")
            # model.renderer.net.heads.multiviewhead.register_forward_hook(hook_fn_forward_heads)

        timing["t_to_gpu"] = time.time() - _start_time

        model.train()

        _start_time = time.time()

        with autocast(enabled=with_amp):
            data = model(
                data
            )  ## Forward pass: model == BTSWrapper(nn.Module) or BTSWrapperOverfit(BTSWrapper)  ## data has 8 views for kitti360

        # if model.renderer.net.__class__.__name__ == "BTSNet":

        # calculate the loss based on data["head_outputs"], convert to tensors
        if model.renderer.net.__class__.__name__ == "MVBTSNet" and (requires_pgt != 0):
            data["invalid_features"] = torch.cat(invalid_features, dim=0)
            data["head_outputs"] = {name: torch.cat(predictions, dim=0) for name, predictions in head_outputs.items()}
            invalid_features = []
            head_outputs = {name: [] for name, _ in model.renderer.net.heads.items()}
            # data["head_outputs"] = {name: predictions for name, predictions in head_outputs.items()}

        timing["t_forward"] = time.time() - _start_time

        _start_time = time.time()
        loss, loss_metrics = criterion(data)
        timing["t_loss"] = time.time() - _start_time

        # for handle in hook_fw_handles:  handle.remove()   ## remove hooks for safety

        _start_time = time.time()
        optimizer.zero_grad()
        scaler.scale(loss).backward()  ## make same scale for gradients. Note: it's not ignite built-in func. (c.f. https://wandb.ai/wandb_fc/tips/reports/How-To-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5)
        # scaler.scale(loss).backward(retain_graph=True)       ## make same scale for gradients. Note: it's not ignite built-in func. (c.f. https://wandb.ai/wandb_fc/tips/reports/How-To-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5)
        scaler.step(optimizer)
        # bwd_hook_debug_init_state = {param.clone() for param in model.renderer.net.heads.multiviewhead.attn_layers.layers[0].linear1.parameters()}
        scaler.update()
        # bwd_hook_debug_cmp_state = {param for param in model.renderer.net.heads.multiviewhead.attn_layers.layers[0].linear1.parameters()}
        # print("cmp", bwd_hook_debug_init_state == cmp_state)
        timing["t_backward"] = time.time() - _start_time

        return {"output": data, "loss_dict": loss_metrics, "timings_dict": timing, "metrics_dict": {}}

    trainer = Engine(train_step)
    trainer.logger = logger

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    to_save = {"trainer": trainer, "model": model, "lr_scheduler": lr_scheduler}
    to_save_new_model = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save_new_model,
        save_every_iters=config["checkpoint_every"],
        save_handler=get_save_handler(config),
        lr_scheduler=lr_scheduler,
        output_names=None,
        with_pbars=False,
        clear_cuda_cache=False,
        log_every_iters=config.get("log_every_iters", 100),
        n_saved=10,
    )

    resume_from = config["resume_from"]
    if resume_from is not None:
        new_model_architecture = config.get("new_model_architecture", False)
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), f"__Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        logger.info(f"__Resume from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        if new_model_architecture:
            Checkpoint.load_objects(to_load=to_save_new_model, checkpoint=checkpoint, strict=False)
        else:
            Checkpoint.load_objects(
                to_load=to_save, checkpoint=checkpoint, strict=False
            )  ## !strict := matching parameters only done mis match ML != DFT

    return trainer


def create_evaluator(model, metrics, criterion, config, tag="val"):
    with_amp = config["with_amp"]
    device = idist.device()

    @torch.no_grad()
    def evaluate_step(engine: Engine, data):
        model.eval()
        if "t__get_item__" in data:
            timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
        else:
            timing = {}

        data = to(data, device)

        with autocast(enabled=with_amp):
            data = model(data)  ### BTSWrapperOverfit

        for name in metrics.keys():
            data[name] = data[
                name
            ].mean()  ## origin # if 'abs_rel' in data:   data[name] = data[name].mean()    ## key error handler as overfitting
            ## data.keys() == ['imgs', 'projs', 'poses', 'depths', '3d_bboxes', 'segs', 't__get_item__', 'index', 'fine', 'coarse', 'rgb_gt', 'rays', 'z_near', 'z_far']
        if criterion is not None:  ## !
            loss, loss_metrics = criterion(data)
        else:
            loss_metrics = {}

        return {"output": data, "loss_dict": loss_metrics, "timings_dict": timing, "metrics_dict": {}}

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    if idist.get_rank() == 0 and (not config.get("with_clearml", False)):
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)

    return evaluator


def get_save_handler(config):
    print("=======SAVING========")
    print(config["output_path"])
    return DiskSaver(config["output_path"], require_empty=False)


class MetricLoggingHandler(BaseHandler):
    def __init__(
        self, tag, optimizer=None, log_loss=True, log_metrics=True, log_timings=True, global_step_transform=None
    ):
        self.tag = tag
        self.optimizer = optimizer
        self.log_loss = log_loss
        self.log_metrics = log_metrics
        self.log_timings = log_timings
        self.gst = global_step_transform
        super(MetricLoggingHandler, self).__init__()

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, EventEnum]):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'MetricLoggingHandler' works only with TensorboardLogger")

        if self.gst is None:
            gst = global_step_from_engine(engine)
        else:
            gst = self.gst
        global_step = gst(engine, event_name)  # type: ignore[misc]

        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        writer = logger.writer

        # Optimizer parameters
        if self.optimizer is not None:
            params = {k: float(param_group["lr"]) for k, param_group in enumerate(self.optimizer.param_groups)}

            for k, param in params.items():
                writer.add_scalar(f"lr-{self.tag}/{k}", param, global_step)

        if self.log_loss:
            # Plot losses
            loss_dict = engine.state.output["loss_dict"]
            for k, v in loss_dict.items():
                if not isinstance(v, (float, int)):
                    print(f"{k}: {type(v)}")
                writer.add_scalar(f"loss-{self.tag}/{k}", v, global_step)

        if self.log_metrics:
            # Plot metrics
            metrics_dict = engine.state.metrics
            metrics_dict_custom = engine.state.output["metrics_dict"]

            for k, v in metrics_dict.items():
                if not isinstance(v, (float, int)):
                    print(f"{k}: {type(v)}")
                writer.add_scalar(f"metrics-{self.tag}/{k}", v, global_step)
            for k, v in metrics_dict_custom.items():
                if not isinstance(v, (float, int)):
                    print(f"{k}: {type(v)}")
                writer.add_scalar(f"metrics-{self.tag}/{k}", v, global_step)

        if self.log_timings:
            # Plot timings
            timings_dict = engine.state.times
            timings_dict_custom = engine.state.output["timings_dict"]
            for k, v in timings_dict.items():
                if k == "COMPLETED":
                    continue
                writer.add_scalar(f"timing-{self.tag}/{k}", v, global_step)
            for k, v in timings_dict_custom.items():
                writer.add_scalar(f"timing-{self.tag}/{k}", v, global_step)


class IterationTimeHandler:
    def __init__(self):
        self._start_time = None

    def start_iteration(self, engine):
        self._start_time = time.time()

    def end_iteration(self, engine):
        if self._start_time is None:
            t_diff = 0
            iters_per_sec = 0
        else:
            t_diff = max(time.time() - self._start_time, 1e-6)
            iters_per_sec = 1 / t_diff
        if not hasattr(engine.state, "times"):
            engine.state.times = {}
        else:
            engine.state.times["secs_per_iter"] = t_diff
            engine.state.times["iters_per_sec"] = iters_per_sec


class DataloaderTimeHandler:
    def __init__(self):
        self._start_time = None

    def start_get_batch(self, engine):
        self._start_time = time.time()

    def end_get_batch(self, engine):
        if self._start_time is None:
            t_diff = 0
            iters_per_sec = 0
        else:
            t_diff = max(time.time() - self._start_time, 1e-6)
            iters_per_sec = 1 / t_diff
        if not hasattr(engine.state, "times"):
            engine.state.times = {}
        else:
            engine.state.times["get_batch_secs"] = t_diff


class VisualizationHandler(BaseHandler):
    def __init__(self, tag, visualizer, global_step_transform=None):
        self.tag = tag
        self.visualizer = visualizer
        self.gst = global_step_transform
        super(VisualizationHandler, self).__init__()

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, EventEnum]) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'VisualizationHandler' works only with TensorboardLogger")

        if self.gst is None:
            gst = global_step_from_engine(engine)
        else:
            gst = self.gst
        global_step = gst(engine, event_name)  # type: ignore[misc]

        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        self.visualizer(engine, logger, global_step, self.tag)
