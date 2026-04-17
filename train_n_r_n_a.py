import sys
sys.path.append("../utils/")
import torch
import numpy as np
import torch.nn as nn
from dataset import ToTensor, EEGDataset2DLeftRight, EEGAugmentor, EEGDatasetLeftFeet, EEGDatasetRightFeet
from torchvision import transforms
from utility import train_validate_split_subjects, samples_per_class
from torch.utils.data import DataLoader, sampler
import torch.optim as optim
from snn_n_r_n_a import WrapCUBASpikingCNN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from pathlib import Path
import logging

# Setup a module-level logger that writes to both stdout and a file in training_outputs/
_OUT_DIR = Path.cwd() / "training_outputs"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _OUT_DIR / "train.log"
logger = logging.getLogger('eeg_train')
logger.setLevel(logging.INFO)
if not logger.handlers:
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(_LOG_FILE, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)

def set_seed(seed: int):
    """Set random seeds for reproducibility across random, numpy and torch."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    # make cudnn deterministic (may reduce performance)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def test_accuracy(network, test_loader, device):
    """Compute overall and per-class accuracy on test_loader.
    Returns: (overall_acc (float), class_acc (np.ndarray))
    """
    network.eval()
    with torch.no_grad():
        class_correct = None
        class_total = None
        all_correct = 0
        all_total = 0
        for data in test_loader:
            eeg_data, label = data
            eeg_data = eeg_data.to(device)
            output = network(eeg_data)
            _, predicted = torch.max(output, 1)
            pred_np = predicted.cpu().numpy()
            label_np = label.cpu().numpy()  # ensure label on CPU
            if class_correct is None:
                n_classes = int(label_np.max()) + 1
                class_correct = np.zeros(n_classes, dtype=np.int64)
                class_total = np.zeros(n_classes, dtype=np.int64)
            eq = np.equal(pred_np, label_np)
            for i in range(label_np.shape[0]):
                la = int(label_np[i])
                class_total[la] += 1
                if bool(eq[i]):
                    class_correct[la] += 1
                    all_correct += 1
                all_total += 1
    overall = float(all_correct) / float(all_total) if all_total > 0 else 0.0
    class_acc = (class_correct.astype(float) / (class_total + 1e-8)) if class_total is not None else np.zeros(1)
    network.train()
    return overall, class_acc


def train_network(dataset=EEGDatasetLeftFeet, network=WrapCUBASpikingCNN,
                  dataset_kwargs=None, spike_ts=160, param_list=None,
                  validate_subject_list=None, lr=None, weight_decays=None,
                  batch_size=64, epoch=20, record_neuron=None, seed: int = None,
                  scheduler_type: str = 'plateau', onecycle_max_lr: float = None, onecycle_pct_start: float = 0.3):
    """Train the network.
    This cleaned version preserves logging but does NOT use AMP, schedulers, weighted sampling,
    gradient clipping, or best-model checkpointing.
    """
    # Defaults for mutable args
    if dataset_kwargs is None:
        dataset_kwargs = {}
    if param_list is None:
        param_list = []
    if validate_subject_list is None:
        # default validation subjects (same as original script intent)
        validate_subject_list = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    if lr is None:
        lr = [0.0001, 0.00001, 2 * 0.0001]
    if weight_decays is None:
        weight_decays = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds if provided for reproducibility
    if seed is not None:
        set_seed(int(seed))
        logger.info('Set random seed to: %d', int(seed))

    # Setup network
    net = network(spike_ts, device, param_list=param_list, record_neuron=record_neuron)
    net = nn.DataParallel(net.to(device))
    net.train()

    # Dataset and transforms (train/val transforms should be provided at module-level or in dataset_kwargs)
    train_ds_kwargs = dataset_kwargs.copy()
    train_ds_kwargs.setdefault("transform", ToTensor())
    train_ds = dataset(**train_ds_kwargs)

    val_ds_kwargs = dataset_kwargs.copy()
    val_ds_kwargs.setdefault("transform", ToTensor())
    val_ds = dataset(**val_ds_kwargs)

    train_indices, val_indices = train_validate_split_subjects(train_ds, validate_subject_list)

    logger.info("Training Samples per Class:")
    samples_per_class(train_ds.label[train_indices])
    logger.info("Validate Samples per Class:")
    samples_per_class(val_ds.label[val_indices])

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    val_sampler = sampler.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=4, pin_memory=True)

    # Optimizer and loss (simple baseline)
    # Keep parameter grouping for neuron/ts params if model exposes them
    try:
        decays = ['module.snn.c1_vdecay', 'module.snn.c2_vdecay', 'module.snn.c3_vdecay',
                  'module.snn.tc1_vdecay', 'module.snn.tc1_cdecay', 'module.snn.r1_vdecay',
                  'module.snn.f1_vdecay', 'module.snn.c1_cdecay', 'module.snn.c2_cdecay',
                  'module.snn.c3_cdecay', 'module.snn.r1_cdecay', 'module.snn.f1_cdecay']
        ts_weights = ['module.snn.ts_weights']
        decay_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in decays, net.named_parameters()))))
        ts_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in ts_weights, net.named_parameters()))))
        weights = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in decays + ts_weights, net.named_parameters()))))
        optimizer = optim.Adam([{'params': weights}, {'params': decay_params, 'lr': lr[0]}, {'params': ts_params, 'lr': lr[1]}], lr=lr[2])
    except Exception:
        optimizer = optim.Adam(net.parameters(), lr=lr[2] if len(lr) > 2 else 1e-4)

    criterion = nn.CrossEntropyLoss()

    # Setup learning-rate scheduler (dynamic LR)
    scheduler = None
    scheduler_step_per = None
    try:
        if scheduler_type == 'onecycle':
            # OneCycleLR requires total_steps = epochs * iterations_per_epoch
            total_steps = int(epoch * len(train_loader)) if len(train_loader) > 0 else None
            max_lr = onecycle_max_lr if onecycle_max_lr is not None else (lr[2] * 10 if len(lr) > 2 else 1e-3)
            if total_steps is None:
                logger.warning('OneCycleLR requested but train_loader length unknown; falling back to no scheduler')
            else:
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps,
                                                          pct_start=onecycle_pct_start, anneal_strategy='cos')
                scheduler_step_per = 'iter'
                logger.info('Using OneCycleLR max_lr=%.6g total_steps=%d pct_start=%.2f', max_lr, total_steps, onecycle_pct_start)
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                                             min_lr=1e-8, verbose=True)
            scheduler_step_per = 'epoch'
            logger.info('Using ReduceLROnPlateau scheduler')
        else:
            scheduler = None
            scheduler_step_per = None
            logger.info('No LR scheduler will be used (scheduler_type=%s)', scheduler_type)
    except Exception as ex:
        logger.exception('Failed to create scheduler: %s', ex)

    # keep lr history (per-iteration) so we can inspect later
    lr_history = []

    # Training loop (simple)
    epoch_accs = []
    epoch_losses = []
    epoch_class_accs = []

    for e in range(epoch):
        running_loss = 0.0
        iters = 0
        for i, data in enumerate(train_loader, 0):
            eeg_data, label = data
            eeg_data = eeg_data.to(device)  # type: ignore[attr-defined]
            label = label.to(device)  # type: ignore[attr-defined]

            optimizer.zero_grad()
            output = net(eeg_data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # scheduler step per-iteration (for OneCycleLR)
            if scheduler is not None and scheduler_step_per == 'iter':
                try:
                    scheduler.step()
                except Exception:
                    pass

            # record current LR
            try:
                lr_history.append(optimizer.param_groups[0]['lr'])
            except Exception:
                lr_history.append(0.0)

            running_loss += loss.item()
            iters += 1

        net.eval()
        acc, class_acc = test_accuracy(net, val_loader, device)
        net.train()

        avg_loss = running_loss / max(1, iters)
        epoch_accs.append(float(acc))
        epoch_losses.append(float(avg_loss))
        epoch_class_accs.append(class_acc.copy())

        logger.info('Epoch: %d, Loss: %.6f', e, avg_loss)
        logger.info('Overall validation Accuracy after Epoch %d: %.3f %%', e, acc * 100)
        try:
            logger.info('Accuracy for class 0: %.3f %%, 1: %.3f %%', float(class_acc[0]) * 100, float(class_acc[1]) * 100)
        except Exception:
            logger.info('Class-wise accuracy: %s', class_acc)

        # Scheduler step on validation accuracy (for ReduceLROnPlateau)
        if scheduler is not None and scheduler_step_per == 'epoch':
            try:
                scheduler.step(acc)
            except Exception:
                pass

        # log a snapshot LR for this epoch
        try:
            logger.info('Current LR after epoch %d: %.6g', e, optimizer.param_groups[0]['lr'])
        except Exception:
            pass

    # Save metrics (keep this - non-strategy logging artifact)
    try:
        out_dir = _OUT_DIR
        acc_csv_path = out_dir / "accuracy_per_epoch.csv"
        np.savetxt(acc_csv_path, np.array(epoch_accs), delimiter=',')
        np.save(out_dir / "accuracy_per_epoch.npy", np.array(epoch_accs))

        np.savetxt(out_dir / "loss_per_epoch.csv", np.array(epoch_losses), delimiter=',')
        np.save(out_dir / "loss_per_epoch.npy", np.array(epoch_losses))

        # simple plot
        plt.figure(figsize=(8, 5))
        epochs_arr = np.arange(1, len(epoch_accs) + 1)
        plt.plot(epochs_arr, epoch_accs, marker='o', label='Validation Accuracy')
        plt.plot(epochs_arr, epoch_losses, marker='x', label='Training Loss')
        plt.title('Validation Accuracy and Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_path = out_dir / 'accuracy_loss_per_epoch.png'
        plt.savefig(plot_path)
        plt.close()

        logger.info('Saved accuracy curve to: %s', str(plot_path))
        logger.info('Saved raw accuracy CSV to: %s', str(acc_csv_path))
        # Save LR history too
        try:
            lr_csv = out_dir / 'lr_history.csv'
            np.savetxt(lr_csv, np.array(lr_history), delimiter=',')
            np.save(out_dir / 'lr_history.npy', np.array(lr_history))
            logger.info('Saved LR history to: %s', str(lr_csv))
        except Exception:
            logger.exception('Failed to save lr history')
    except Exception:
        logger.exception('Failed to save accuracy plot or metrics')

    return net


if __name__ == "__main__":
    # Default transforms used when __main__ runs
    train_transform = transforms.Compose([
        EEGAugmentor(time_shift_range=3, noise_std=0.02, prob=0.7),
        ToTensor()
    ])
    val_transform = ToTensor()

    DATASET = EEGDatasetLeftFeet
    USE_IMAGERY = False
    ds_params = {"base_route": "../utils/eegmmidb_slice_norm/",
                 "subject_id_list": [i + 1 for i in range(109) if not (i + 1 in [88, 89, 92, 100, 104, 106])],
                 "start_ts": 0,
                 "end_ts": 161,
                 "window_ts": 160,
                 "overlap_ts": 0,
                 "use_imagery": USE_IMAGERY,
                 "transform": ToTensor()}

    SPIKE_TS = 160
    BATCH_SIZE = 64
    WT_LR = 0.0001
    TS_LR = 0.0001
    NEURON_LR = 0.0001

    PARAM_LIST = [0.1, 0.1, 0.1, 0.3, 0.01, 0.1, 0.01]

    val_list = [[i + 1 for i in range(10)],
                [i + 11 for i in range(10)],
                [i + 21 for i in range(10)],
                [i + 31 for i in range(10)],
                [i + 41 for i in range(10)],
                [i + 51 for i in range(10)],
                [i + 61 for i in range(10)],
                [i + 71 for i in range(10)],
                [81, 82, 83, 84, 85, 86, 87, 90, 91, 93],
                [94, 95, 96, 97, 98, 99, 101, 102, 103, 105]]

    # Set a seed for reproducible runs (change or set to None to disable)
    SEED =4
    train_network(dataset_kwargs=ds_params, spike_ts=SPIKE_TS, param_list=PARAM_LIST, dataset=DATASET,
                  batch_size=BATCH_SIZE, epoch=50, lr=[NEURON_LR, TS_LR, WT_LR], weight_decays=[2e-6, 4e-6, 1e-6],
                  validate_subject_list=val_list[0], record_neuron=(0, 0, 0), seed=SEED)
