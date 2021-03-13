import argparse
import os
import shutil
import time
import random

def parse_args(args):
    cmd = argparse.ArgumentParser(args[0], conflict_handler="resolve")
    cmd.add_argument("--seed", default=42, type=int, help="The random seed")
    cmd.add_argument("--device", default="cuda", type=str, help="cuda or cpu")
    cmd.add_argument("--train_path", help="Path to training file")
    cmd.add_argument("--test_path", help="Path to test file")
    cmd.add_argument("--max_grad_norm", default=1, type=float, help="gradient clipping parameter")
    cmd.add_argument("--model_path", default=None, help="actual path to save model")
    cmd.add_argument("--logfile", default="log.txt.gz")
    cmd.add_argument("--dev_size", default=80, type=int, help="Number of sentences to use as dev data")
    cmd.add_argument("--batch_size", type=int, default=16, help="Number of sentences in each batch")
    cmd.add_argument("--dev_batch_size", type=int, default=20, help="Number of sentences in each dev batch")
    cmd.add_argument("--dropout_1", default=0.1, type=float, help="First dropout rate")
    cmd.add_argument("--hidden_dim", default=385, type=int, help="Number of hidden units")
    cmd.add_argument("--activation", default="relu", choices=["relu", "gelu", "elu", "sigmoid", "tanh"], help="Activation function: valid options=[relu, gelu, elu, sigmoid, tanh]")
    cmd.add_argument("--dropout_2", default=0.1, type=float, help="Second dropout rate")
    cmd.add_argument("--learning_rate", default=1e-5, type=float, help="Second dropout rate")
    cmd.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay parameter for AdamW")
    cmd.add_argument("--num_warmup_prop", default=0.1, type=float, help="Proportion of warmup steps")
    cmd.add_argument("--num_train_epochs", default=10, type=int, help="Number of training epochs")
    cmd.add_argument("--evaluate_every", default=1, type=int, help="Evaluate every n epochs")
    cmd.add_argument("--dep_var", required=True, choices=["nfix", "ffd", "gpt", "trt", "fprop"], help="Dependent variable: valid options=[nfix, ffd, gpt, trt, fprop]")
    cmd.add_argument("--pretrained_model", required=True, choices=["roberta-base", "roberta-large"], help="Pretrained model: valid options=[roberta-base, roberta-large]")
    cmd.add_argument("--freeze_roberta", default=False, action="store_true", help="Freeze RoBERTa weights")
    cmd.add_argument("--ablate_wlen", default=False, action="store_true", help="Ablate 'word length' feature")
    cmd.add_argument("--ablate_prop", default=False, action="store_true", help="Ablate 'proportion processed' feature")
    cmd.add_argument("--checkpoint", default="", help="Model file to continue training/conduct evaluation with")

    opt = cmd.parse_args(args[2:])

    if opt.model_path is None:
        opt.model_path = os.path.join("output", "{}_{}_epoch{}_batch{}_lr{}".format(opt.pretrained_model, opt.dep_var, opt.num_train_epochs, opt.batch_size, opt.learning_rate))
        # suspends operation for given number of seconds
        time.sleep(random.uniform(0, 5))
        for i in range(100):
            checking_path = opt.model_path+"_"+str(i)
            if not os.path.exists(checking_path):
                opt.model_path = checking_path
                break
    else:
        if os.path.exists(opt.model_path):
            shutil.rmtree(opt.model_path)
    os.makedirs(opt.model_path)

    arg_file = os.path.join(opt.model_path, "args.txt")
    with open(arg_file, "w") as afh:
        print(vars(opt), file=afh)

    return opt