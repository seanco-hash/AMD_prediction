import os
import torch
# from TimeSformer.timesformer.utils.checkpoint import sub_to_normal_bn

# TODO change the path
# TODO change the model checkpoint name according to the cfg flie
path_to_checkpoint = "/cs/labs/dina/seanco/hadassah/dl_project/AMD/models/checkpoints/{}.cp"


def is_checkpoint_exists(cfg):
    file_name = f"{cfg.MODEL.CHECKPOINT_NAME}_{cfg.SOLVER.OPTIMIZING_METHOD}_{cfg.SOLVER.BASE_LR}"
    return os.path.isfile(path_to_checkpoint.format(file_name))


def load_check_point(model, cfg, optimizer):
    file_name = f"{cfg.MODEL.CHECKPOINT_NAME}_{cfg.SOLVER.OPTIMIZING_METHOD}_{cfg.SOLVER.BASE_LR}"
    cp = path_to_checkpoint.format(file_name)
    checkpoint = torch.load(cp)

    cfg = checkpoint["cfg"]
    epoch = checkpoint["epoch"]
    epochs_times = checkpoint["epochs_times"]
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    train_loss = checkpoint["train_loss"]
    eval_loss = checkpoint["val_loss"]
    val_acc = checkpoint["val_acc"]
    acc_per_class = checkpoint["acc_per_class"]
    writer_log_dir = checkpoint["writer_log_dir"]

    return (
        cfg,
        epoch,
        epochs_times,
        model,
        optimizer,
        train_loss,
        eval_loss,
        val_acc,
        acc_per_class,
        writer_log_dir
    )


def save_check_point(cfg, model, optimizer, epoch, epochs_times, train_loss, val_loss, val_acc,
                     acc_per_class, writer_log_dir, i):
    file_name = f"{cfg.MODEL.CHECKPOINT_NAME}_{str(i)}"
    print(f"Saving model: {file_name}")

    # Record the state
    checkpoint = {
        # "cfg": cfg.dump(),
        "cfg": cfg,
        "epoch": epoch,
        "epochs_times": epochs_times,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "acc_per_class": acc_per_class,
        "writer_log_dir": writer_log_dir
    }

    with open(path_to_checkpoint.format(file_name), "wb") as f:
        torch.save(checkpoint, f)

    return path_to_checkpoint
