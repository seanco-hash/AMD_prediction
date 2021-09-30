import argparse
from datasets.oct_dataset import OCTDataset
from models.model import create_model
from models.optimizer import construct_optimizer
from models.optimizer import update_lr
import numpy as np
import os
import time
from timesformer.config.defaults import get_cfg
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import utils.checkpoint as checkpoint
import utils.visualizer
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_dataset_distribution(loader):
    """
    Calculates dataset mean and std of pixel values.
    For use before training, not part of the training.
    :param loader: Dataloader object
    :return:
    """
    try:
        u = 0.0782
        real_std = 0.
        _sum = 0.
        _sum_sq = 0.
        n_frames = 0
        nb_batches = 0.
        for data in loader:
            batch = data[0]
            # mean += batch[:, :, 0, :, :].mean()
            _sum += batch[:, :, 0, :, :].sum()
            # _sum_sq += (batch[:, :, 0, :, :] ** 2).sum()
            real_std += ((batch[:, :, 0, :, :] - u) ** 2).sum()
            nb_batches += 1
            n_frames += batch.size()[1]
            if nb_batches % 100 == 0:
                print(nb_batches)

        n_pixels = n_frames * 254 * 254
        tot_mean = _sum / n_pixels
        real_std = np.sqrt(real_std / n_pixels)
        print('data mean: ', tot_mean)
        # var = (_sum_sq / n_pixels) - (tot_mean ** 2)
        # tot_std = np.sqrt(var)
        print('data std:', real_std)
        # print('epochs means: ', means)
        # print('epochs stds: ', stds)
        exit(0)
    except Exception as e:
        print(e)


def load_config(args):
    """
    Loads config file
    :param args:
    :return:
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)  # Allow new attributes

    # Load config from cfg.
    cfg.merge_from_file(args.cfg_file)

    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr

    if args.epochs is not None:
        cfg.TRAIN.EPOCHS = args.epochs

    return cfg


def get_criteria(cfg, data=None):
    """
    Creates loss function according to cfg.
    Can calculate class weights from Dataset object
    :param cfg:
    :param data:
    :return:
    """
    if cfg.MODEL.LOSS_CLASS_WEIGHT and data is not None:
        weights = data.get_labels_distribution()
    else:
        weights = None
    if cfg.MODEL.NUM_CLASSES > 2:
        if weights is not None:
            weights = weights / np.sum(weights)
            weights = 1 / weights
            weights = torch.FloatTensor(weights).cuda()
        return [torch.nn.CrossEntropyLoss(weight=weights)]
    else:
        if weights is not None:
            weight = (weights[0] / weights[1])
            weights = torch.as_tensor(weight, dtype=torch.float)
        return [torch.nn.BCEWithLogitsLoss(pos_weight=weights)]


def get_targets(labels):
    """
    Reshape mini-batch labels in order to compute the loss.
    :param labels:
    :return:
    """
    return [labels]


def train_epoch(model, train_loader, criteria, optimizer, epoch, train_loss, epochs_times,
                num_features_to_predict, writer, n_classes):
    """
    Trains single epoch
    :param model:
    :param train_loader:
    :param criteria:
    :param optimizer:
    :param epoch: current epoch num
    :param train_loss: list of train loss at end of epoch
    :param epochs_times:
    :param num_features_to_predict:
    :param writer: Tensorboard SummaryWriter object
    :param n_classes: number of classes to predict
    :return:
    """
    model.train()

    start_time = time.time()
    running_loss = 0
    losses = np.zeros(num_features_to_predict, dtype=np.float128)

    # Go over all the data, tqdm - progress bar
    # data_loader = tqdm(train_loader, position=0, leave=True)
    data_loader = train_loader
    for i, data in enumerate(data_loader, 0):
        # Get the current batch
        inputs, labels = data
        inputs = inputs.permute([0, 2, 1, 3, 4])
        labels = get_targets(labels)
        inputs = inputs.to(device)
        # Zero the parameter gradients.
        optimizer.zero_grad()

        # Forward + Backward + Optimize.
        outputs = model(inputs)

        # Calculate loss for each output.
        loss = 0
        for idx, (output, target, criterion) in enumerate(zip(outputs, labels, criteria)):
            target = target.to(device)
            if n_classes == 2:
                target = torch.reshape(target, output.shape)
                target = target.float()
            cur_loss = criterion(output, target)
            loss = loss + cur_loss
            losses[idx] += cur_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # For prgress bar
        # data_loader.set_postfix({'loss': loss.item()})
        if (i + 1) % 20 == 0:
            running_loss /= 20
            print(f"[Train] Epoch: {epoch} Batch: {i+1} Loss: {running_loss:.3f}")
            running_loss = 0.0

    losses /= len(train_loader)
    writer.add_scalar("Loss/train", losses[0].item(), epoch)
    train_loss.append(losses)  # Add the epoch loss to the list of losses
    epochs_times.append(start_time - time.time())
    print("Finished epoch:", epoch)


def plot_roc_curve(fpr, tpr, auc, cfg, i=0):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    out_dir = '/cs/labs/dina/seanco/hadassah/dl_project/AMD/plots/'
    fig_name = out_dir + cfg.MODEL.CHECKPOINT_NAME + '_roc_' + str(i) + '.png'
    plt.show()

    plt.savefig(fig_name)


def calc_precision_recall_roc(confusion_matrix, n_classes, pred_prob, true_y, cfg, i):
    if n_classes > 2:
        return 0, 0, 0
    tp = confusion_matrix[1, 1]
    fn = confusion_matrix[1, 0]
    fp = confusion_matrix[0, 1]
    tn = confusion_matrix[0, 0]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fpr, tpr, thresholds = metrics.roc_curve(true_y, pred_prob)
    auc = metrics.roc_auc_score(true_y, pred_prob)
    plot_roc_curve(fpr, tpr, auc, cfg, i)
    return auc, recall, precision


def multi_acc(y_pred, y_test, confusion_matrix=None):
    """
    Calculate accuracy of mini-batch for multi-class calssification
    :param y_pred: predicted probabilities
    :param y_test: ground true
    :param confusion_matrix:
    :return:
    """
    y_pred_softmax = torch.softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    if confusion_matrix is not None:
        for t, p in zip(y_test, y_pred_tags):
            confusion_matrix[t.int(), p.int()] += 1
    return correct_pred.sum() / len(y_test), y_pred_softmax


def binary_acc(y_pred, y_test, confusion_matrix=None, probability_matrix=None):
    """
    Compute the accuracy for binary classification mini-batch.
    :param y_pred:
    :param y_test:
    :param confusion_matrix:
    :param probability_matrix:
    :return:
    """
    y_pred_prob = torch.sigmoid(y_pred)
    y_pred_tag = torch.round(y_pred_prob)
    correct_results = (y_pred_tag == y_test).float()
    if confusion_matrix is not None:
        for i, (t, p) in enumerate(zip(y_test, y_pred_tag)):
            confusion_matrix[t.long(), p.long()] += 1
            if probability_matrix is not None:
                probability_matrix[t.long()][p.long()].append(y_pred_prob[i].item())
    return correct_results.sum() / len(y_test), y_pred_prob


def calculate_acc(output, target, confusion_matrix=None, probability_matrix=None):
    # Calculate the accuracy of single output.
    if target.ndim == 1:
        return multi_acc(output, target, confusion_matrix)
    return binary_acc(output, target, confusion_matrix, probability_matrix)


def print_eval_final_statistics(losses, val_acc, epoch):
    # Print final statistics of the evaluate phase.
    print(f"[Eval] Epoch: {epoch} Avg loss: {np.mean(losses)} Avg accuracy: {np.mean(val_acc)}")
    loss_msg = "[Eval] features lossess \n"
    acc_msg = "[Eval] features accuracies \n"
    for i in range(len(losses)):
        loss_msg += f"feature {i} loss: {losses[i]}\n"
        acc_msg += f"feature {i} acc: {val_acc[i]}\n"

    print(loss_msg)
    print(acc_msg + "\n")


def eval_epoch(model, val_loader, criteria, epoch, val_loss, val_acc, num_feat_to_pred, writer, n_classes,
               max_epochs):
    """
    Evaluating performance on validation set
    """
    model.eval()
    with torch.no_grad():
        # Saves all prediction for ROC calculation
        predicted_probabilities = list()
        true_labels = list()
        confusion_matrix = torch.zeros(n_classes, n_classes)
        # Saves stats for histogram od decisions
        probability_matrix = [[[] for i in range(n_classes)] for j in range(n_classes)]
        losses = np.zeros(num_feat_to_pred, dtype=np.float128)
        eval_acc = np.zeros(num_feat_to_pred, dtype=np.float128)

        # data_loader = tqdm(val_loader, position=0, leave=True)
        data_loader = val_loader
        for data in data_loader:
            # Get the current batch
            inputs, labels = data
            inputs = inputs.permute([0, 2, 1, 3, 4]).to(device)
            labels = get_targets(labels)
            outputs = model(inputs)

            loss = 0
            for idx, (output, target, criterion) in enumerate(zip(outputs, labels, criteria)):
                target = target.to(device)
                if n_classes == 2:
                    target = torch.reshape(target, output.shape)
                    target = target.float()
                cur_loss = criterion(output, target)
                losses[idx] += cur_loss
                # Calculates ROC curve if last epoch
                if epoch == max_epochs:
                    acc, cur_predicted_probabilities = calculate_acc(output, target, confusion_matrix,
                                                               probability_matrix)
                    eval_acc[idx] += acc
                    predicted_probabilities.append(np.asarray(cur_predicted_probabilities.to('cpu')).flatten())
                    true_labels.append(np.asarray(target.to('cpu')).flatten())
                else:
                    acc, _ = calculate_acc(output, target, confusion_matrix, None)
                    eval_acc[idx] += acc
                loss += cur_loss

            # Print statistics.
            # data_loader.set_postfix({'loss': loss.item(), "Epoch": epoch})

        losses = losses / len(val_loader)
        eval_acc = eval_acc / len(val_loader)
        writer.add_scalar("Loss / Validation", losses[0].item(), epoch)
        writer.add_scalar("Accuracy / Validation", eval_acc[0], epoch)

        # Print final statistics.
        print_eval_final_statistics(losses, eval_acc, epoch)
        print(confusion_matrix.int())
        per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
        print('per class acc: ', per_class_acc)
        if n_classes == 3:
            writer.add_scalars('Per-Class-Accuracy / Validation', {'0': per_class_acc[0], '1': per_class_acc[1],
                                                                   '2': per_class_acc[2]}, epoch)
        elif n_classes == 2:
            # Add statistics to tensorboard
            writer.add_scalars('Per-Class-Accuracy / Validation',
                               {'0': per_class_acc[0], '1': per_class_acc[1]}, epoch)
            if epoch == max_epochs:
                writer.add_histogram('Prediction Probability TN', np.asarray(probability_matrix[0][0]), epoch)
                writer.add_histogram('Prediction Probability FP', np.asarray(probability_matrix[0][1]), epoch)
                writer.add_histogram('Prediction Probability FN', np.asarray(probability_matrix[1][0]), epoch)
                writer.add_histogram('Prediction Probability TP', np.asarray(probability_matrix[1][1]), epoch)
                predicted_probabilities = np.concatenate(predicted_probabilities)
                true_labels = np.concatenate(true_labels)

        val_loss.append(losses)
        val_acc.append(eval_acc)
        return per_class_acc, predicted_probabilities, true_labels, confusion_matrix


def plot_loss_acc(train_loss, val_loss, val_acc, cfg):
    # Get all classifications losses.
    val_acc = np.asarray(val_acc)
    val_loss = np.asarray(val_loss)
    train_loss = np.array(train_loss)
    losses = list()
    losses.append(train_loss)
    losses.append(val_loss)
    legends = utils.visualizer.get_features()[3:4] + ["Val Loss"]
    utils.visualizer.plot_graph(np.arange(train_loss.shape[0]), losses, "Loss as funtion of epochs",
                                "Epoch", "Loss",cfg,  legends)
    utils.visualizer.plot_graph(np.arange(val_acc.shape[0]), val_acc, "Accuracy as funtion of epochs",
                                "Epoch", "Accuracy", cfg)


def write_hparams(writer, cfg, loss, tot_acc, acc_per_class, auc, recall, precision):
    """
    Write hyper parameters into tensorboard
    """
    # pretrained = cfg.MODEL.PRETRAINED_PATH
    # pretrained = pretrained.split('/')[-1]
    hparams_dict = {'optimizer': cfg.SOLVER.OPTIMIZING_METHOD,
                        'base lr': cfg.SOLVER.BASE_LR, 'decay': cfg.SOLVER.WEIGHT_DECAY,
                        'transforms': cfg.DATA_LOADER.TRANSFORMS, 'class weight': cfg.MODEL.LOSS_CLASS_WEIGHT,
                        'finetune percent': cfg.MODEL.FINETUNE_PERCENT,
                        'slice pick': cfg.DATA_LOADER.METHOD,
                        'feature': cfg.TARGET_FEATURE,
                        'model name': cfg.MODEL.MODEL_NAME}
    metrics_dict = {'hparam/loss': loss,
                        'hparam/accuracy': tot_acc,
                        'hparam/acc_class_0': acc_per_class[0],
                        'hparam/acc_class_1': acc_per_class[1],
                        'hparam/AUC': auc,
                        'hparam/recall': recall,
                        'hparam/precision': precision}
    writer.add_hparams(hparams_dict, metrics_dict)
    print(hparams_dict)
    print(metrics_dict)
    print(cfg.TRAIN.EPOCHS)


def train(args, cfg=None, i=0):
    print("Start running with config: ", args.cfg_file)
    # Load config file.
    if cfg is None:
        cfg = load_config(args)

    # Get model.
    model, optim_params = create_model(cfg)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model.to(device)
    # Get optimizer and loss criteria according to the config file.
    optimizer = construct_optimizer(optim_params, cfg)
    epoch = 0
    writer_log_dir = None
    train_loss, val_loss, val_acc, epochs_times = list(), list(), list(), list()

    # Load checkpoint if exists.
    if cfg.MODEL.USE_CHECKPOINT and checkpoint.is_checkpoint_exists(cfg):
        cfg, epoch, epoch_times, model, optimizer, train_loss, val_loss, val_acc, acc_per_class, \
            writer_log_dir = checkpoint.load_check_point(model, cfg, optimizer)
        print(f"Loaded checkpoint for {cfg.MODEL.MODEL_NAME}")
        print_eval_final_statistics(val_loss[-1], val_acc[-1], epoch)
        cfg.TRAIN.EPOCHS = 16
        if epoch > 10:
            epoch = cfg.TRAIN.EPOCHS - 1

    # Get data loaders.
    training_data = OCTDataset(cfg, train=True)
    test_data = OCTDataset(cfg, train=False)
    criteria = get_criteria(cfg, training_data)
    if cfg.TRAIN.DEBUG:
        num_workers = 1
    else:
        # Gets available cpus to parallelize the data loader
        num_workers = len(os.sched_getaffinity(0)) - 1
    train_loader = DataLoader(training_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    # calc_dataset_distribution(train_loader)
    val_loader = DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=num_workers)

    print("Begin Training")
    # Continues tensorboard writing in case resume from checkpoint
    if writer_log_dir is None or epoch > 10:
        writer = SummaryWriter()
    else:
        writer = SummaryWriter(writer_log_dir)
    best_acc = 0
    while epoch < cfg.TRAIN.EPOCHS:
        epoch += 1
        # Train the model for single epoch and update the learning rate.
        try:
            if epoch < cfg.TRAIN.EPOCHS:
                train_epoch(model, train_loader, criteria, optimizer, epoch, train_loss,
                            epochs_times, cfg.MODEL.NUM_FEATURES_TO_PREDICT, writer, cfg.MODEL.NUM_CLASSES)
                update_lr(optimizer, epoch, cfg)

            # Evaluate the model.
            if epoch % cfg.TRAIN.EVAL_PERIOD == 0 or epoch == cfg.TRAIN.EPOCHS:
                acc_per_class, predicted_probabilities, true_labels, confusion_matrix = \
                    eval_epoch(model, val_loader, criteria, epoch, val_loss, val_acc,
                               cfg.MODEL.NUM_FEATURES_TO_PREDICT, writer, cfg.MODEL.NUM_CLASSES, cfg.TRAIN.EPOCHS)
            # Save the model if it performed the best so far
                if val_acc[-1] > best_acc and acc_per_class[0] > 0.72 and acc_per_class[1] > 0.72:
                    best_acc = val_acc[-1]
                    checkpoint.save_check_point(cfg, model, optimizer, epoch, epochs_times, train_loss,
                                                val_loss, val_acc, acc_per_class, writer.get_logdir(), i)
        except Exception as e:
            print(e)
            raise e
        except SystemExit as e:
            print(e)
            raise e

    # plot_loss_acc(train_loss, val_loss, val_acc, cfg)
    auc_roc, recall, precision = calc_precision_recall_roc(confusion_matrix, cfg.MODEL.NUM_CLASSES,
                                              predicted_probabilities, true_labels, cfg, i)
    write_hparams(writer, cfg, train_loss[-1], val_acc[-1], acc_per_class, auc_roc, recall, precision)
    print_eval_final_statistics(val_loss[-1], val_acc[-1], epoch)
    writer.close()


def train_with_hparam_tune(args):
    """
    Train with hyper parameters tune
    """
    i = 0
    cfg = load_config(args)
    if not cfg.TRAIN.HPARAM_TUNE:
        return train(args, cfg)
    base_lr = [0.005, 0.0001]
    steps = [[0, 5, 12, 18], [0, 8, 16]]
    finetune_percent = [1, 0.75, 0.5, 0.25, 0]
    class_weight = [False, True]
    for j, lr in enumerate(base_lr):
        for ft in finetune_percent:
            cfg.SOLVER.BASE_LR = lr
            cfg.SOLVER.STEPS = steps[j]
            # cfg.MODEL.LOSS_CLASS_WEIGHT = weight
            cfg.MODEL.FINETUNE_PERCENT = ft
            i += 1
            train(args, cfg, i)


class TrainingParser(argparse.ArgumentParser):
    """
    Parse arguments
    """
    def __init__(self, **kwargs):
        super(TrainingParser, self).__init__(**kwargs)
        self.add_argument(
            "--cfg",
            dest="cfg_file",
            help="Path to the config file",
            default="/cs/labs/dina/seanco/hadassah/dl_project/AMD/configs/VIT_8x224_simple_run.yaml",
            type=str,
        )
        self.add_argument(
            "--lr",
            help="Base learning rate",
            type=float,
        )
        self.add_argument(
            "--epochs",
            help="Number of epochs",
            type=int
        )

    @staticmethod
    def validate_args(args):
        if args.cfg_file is None or not os.path.isfile(args.cfg_file):
            raise argparse.ArgumentTypeError(f"Invalid config file path: {args.cfg_file}")

    def parse_args(self, args=None, namespace=None):
        """ Parse the input arguments """
        args = super(TrainingParser, self).parse_args(args, namespace)
        return args


def parse_args():
    parser = TrainingParser()
    args = parser.parse_args()
    TrainingParser.validate_args(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    # train(args)
    train_with_hparam_tune(args)
