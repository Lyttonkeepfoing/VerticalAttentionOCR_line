import sys
sys.path.insert(0, '../../..')
from trainer import TrainerLineCTC
from model import Decoder
from basic.models import FCN_Encoder
from torch.optim import Adam
from basic.dataset_manager import OCRDataset
import torch.multiprocessing as mp
import torch


def train_and_test(rank, params):
    params["training_params"]["ddp_rank"] = rank
    model = TrainerLineCTC(params)  # 传参到LineCTC
    model.train()  # 开始训练

    model.params["training_params"]["load_epoch"] = "best"  # load weights giving best CER on valid set
    model.load_model()

    # compute metrics on train, valid and test sets (in eval conditions)
    metrics = ["cer", "wer", "time", "worst_cer"]
    for dataset_name in params["dataset_params"]["datasets"].keys():
        for set_name in ["test", "valid", "train"]:
            model.predict("{}-{}".format(dataset_name, set_name), [(dataset_name, set_name), ], metrics, output=True)


if __name__ == '__main__':
    dataset_name = "IAM" # ["IAM", "RIMES", "READ_2016"]

    params = {
        "dataset_params": {
            "datasets": {
                dataset_name: "./Datasets/format/{}_lines".format(dataset_name),
            },
            "train": {
                "name": "{}-train".format(dataset_name),
                "datasets": [dataset_name, ],
            },
            "valid": {
                "{}-valid".format(dataset_name): [dataset_name, ],
            },
            "dataset_class": OCRDataset,
            "config": {
                "width_divisor": 8,  # Image width will be divided by 8
                "height_divisor": 32,  # Image height will be divided by 32
                "padding_value": 0,  # Image padding value
                "padding_token": 1000,  # Label padding value (None: default value is chosen)
                "charset_mode": "CTC",  # add blank token
                "constraints": ["CTC_line"],  # Padding for CTC requirements if necessary
                "preprocessings": [
                    {
                        "type": "dpi",  # modify image resolution
                        "source": 300,  # from 300 dpi
                        "target": 150,  # to 150 dpi
                    },
                    {
                        "type": "to_RGB",
                        # if grayscale image, produce RGB one (3 channels with same value) otherwise do nothing
                    },
                ],
                # Augmentation techniques to use at training time
                "augmentation": {
                    "dpi": {
                        "proba": 0.2,
                        "min_factor": 0.75,
                        "max_factor": 1.25,
                    },
                    "perspective": {
                        "proba": 0.2,
                        "min_factor": 0,
                        "max_factor": 0.3,
                    },
                    "elastic_distortion": {
                        "proba": 0.2,
                        "max_magnitude": 20,
                        "max_kernel": 3,
                    },
                    "random_transform": {
                        "proba": 0.2,
                        "max_val": 16,
                    },
                    "dilation_erosion": {
                        "proba": 0.2,
                        "min_kernel": 1,
                        "max_kernel": 3,
                        "iterations": 1,
                    },
                    "brightness": {
                        "proba": 0.2,
                        "min_factor": 0.01,
                        "max_factor": 1,
                    },
                    "contrast": {
                        "proba": 0.2,
                        "min_factor": 0.01,
                        "max_factor": 1,
                    },
                    "sign_flipping": {
                        "proba": 0.2,
                    },
                },
            }
        },

        "model_params": {
            # Model classes to use for each module
            "models": {
                "encoder": FCN_Encoder,
                "decoder": Decoder,
            },
            "transfer_learning": None,  # dict : {model_name: [state_dict_name, checkpoint_path, learnable, strict], }
            "input_channels": 3,  # 1 for grayscale images, 3 for RGB ones (or grayscale as RGB)
            "dropout": 0.5,
        # dropout probability for standard dropout (half dropout probability is taken for spatial dropout)
        },

        "training_params": {
            "output_folder": "fcn_iam_line",  # folder names for logs and weigths
            "max_nb_epochs": 5000,  # max number of epochs for the training
            "max_training_time": 3600 * (24 + 23),  # max training time limit (in seconds)
            "load_epoch": "best",  # ["best", "last"], to load weights from best epoch or last trained epoch
            "interval_save_weights": None,  # None: keep best and last only
            "use_ddp": False,  # Use DistributedDataParallel
            "use_apex": False,  # Enable mix-precision with apex package  # apex下不好就不要用了
            "nb_gpu": torch.cuda.device_count(),
            "batch_size": 1,  # mini-batch size per GPU
            "optimizer": {
                "class": Adam,
                "args": {
                    "lr": 0.0001,
                    "amsgrad": False,
                }
            },
            "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
            "eval_on_valid_interval": 2,  # Interval (in epochs) to evaluate during training
            "focus_metric": "cer",  # Metrics to focus on to determine best epoch
            "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
            "set_name_focus_metric": "{}-valid".format(dataset_name),
            "train_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for training
            "eval_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for evaluation on validation set during training
            "force_cpu": False,  # True for debug purposes to run on cpu only
        },
    }

    if params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
        mp.spawn(train_and_test, args=(params,), nprocs=params["training_params"]["nb_gpu"])
        """
        mp.spawn 分布式支持  fn：派生程序入口；
        nprocs: 派生进程个数；
        join: 是否加入同一进程池；
        daemon：是否创建守护进程；
        """
    else:
        train_and_test(0, params)