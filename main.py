import sys, gzip, os
import argparse
import time
import random
import logging
import torch
import math
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer, RobertaForTokenClassification, RobertaModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from model import RobertaForGazePrediction
import preprocess
from model_args import parse_args

def random_seed(seed_value, use_cuda):
    torch.manual_seed(seed_value) # cpu vars
    random.seed(seed_value) # Python
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars

def train():
    opt = parse_args(sys.argv)
    random_seed(opt.seed, use_cuda=opt.device == "cuda")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logfile = gzip.open(os.path.join(opt.model_path, opt.logfile), "wt")
    writer = SummaryWriter(os.path.join(opt.model_path, "tensorboard"), flush_secs=10)
    filehandler = logging.StreamHandler(logfile)
    streamhandler = logging.StreamHandler(sys.stdout)
    handler_list = [filehandler, streamhandler]
    logging.basicConfig(level="INFO", format="%(asctime)s %(message)s",
                        datefmt="%m/%d/%Y %I:%M:%S %p", handlers=handler_list)
    logging.info(opt)
    writer.add_text("args", str(opt))

    assert (opt.device == "cuda" and torch.cuda.is_available()) or opt.device == "cpu"

    all_data = preprocess.preprocess(opt.train_path, opt.pretrained_model)
    train_data, dev_data = preprocess.divide(all_data, opt.dev_size)
    logging.info("Training sentences: {}, training tokens: {}.".format(len(train_data[-1]), sum([len(s) for s in train_data[-1]])))
    logging.info("Dev sentences: {}, dev tokens: {}.".format(len(dev_data[-1]), sum([len(s) for s in dev_data[-1]])))
    logging.info("Evaluate after every {} steps.".format(opt.evaluate_every))

    input_size = {"roberta-base": 768, "roberta-large": 1024}
    roberta = RobertaModel.from_pretrained(opt.pretrained_model)
    tokenizer = RobertaTokenizer.from_pretrained(opt.pretrained_model)
    # model = RobertaForGazePrediction(pretrained=roberta, input_dim=input_size[opt.pretrained_model], dropout_1=opt.dropout_1,
    #                                  hidden_dim=opt.hidden_dim, activation=opt.activation, dropout_2=opt.dropout_2)
    model = RobertaForGazePrediction(pretrained=roberta, input_dim=input_size[opt.pretrained_model],
                                     dropout_1=opt.dropout_1, hidden_dim=int((input_size[opt.pretrained_model]+2)/2), activation=opt.activation,
                                     dropout_2=opt.dropout_2)
    # model.train()
    logging.info(str(model))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    logging.info("Topmost model has {} parameters".format(num_params))

    model = model.to(opt.device)
    optimizer = AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    criterion = MSELoss()

    if opt.freeze_roberta:
        for name, param in model.pretrained_model.named_parameters():
            logging.info("Freezing parameters {}".format(name))
            param.requires_grad = False

    # steps in each epoch * num_train_epochs
    num_train_steps = math.ceil(len(train_data[-1])/opt.batch_size) * opt.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, round(opt.num_warmup_prop*num_train_steps), num_train_steps)
    lowest_mse = 1e8

    sent_strings, sent_first_idx, sent_wlen, sent_prop, sent_nfix, sent_ffd, sent_gpt, sent_trt, sent_fixprop, sent_words = train_data
    dv_dict = {"nfix": sent_nfix, "ffd": sent_ffd, "gpt": sent_gpt, "trt": sent_trt, "fprop": sent_fixprop}
    dv = dv_dict[opt.dep_var]

    d_sent_strings, d_sent_first_idx, d_sent_wlen, d_sent_prop, d_sent_nfix, d_sent_ffd, d_sent_gpt, d_sent_trt, d_sent_fixprop, d_sent_words = dev_data
    d_dv_dict = {"nfix": d_sent_nfix, "ffd": d_sent_ffd, "gpt": d_sent_gpt, "trt": d_sent_trt, "fprop": d_sent_fixprop}
    d_dv = d_dv_dict[opt.dep_var]

    # in_vocab_idx, oov_idx = preprocess.oov_indices(sent_words, d_sent_words)

    start_time = time.time()

    for epoch in range(opt.num_train_epochs):
        # permute training data
        vars = list(zip(sent_strings, sent_first_idx, sent_wlen, sent_prop, dv))
        random.shuffle(vars)
        sent_strings, sent_first_idx, sent_wlen, sent_prop, dv = zip(*vars)
        # print(sent_strings[0])
        model.train()
        train_loss = 0

        for j in range(0, len(sent_strings), opt.batch_size):
            batch_sentences = sent_strings[j:j+opt.batch_size]
            first_idx = sent_first_idx[j:j+opt.batch_size]
            flat_wlen = [wlen for sublist in sent_wlen[j:j+opt.batch_size] for wlen in sublist]
            flat_prop = [prop for sublist in sent_prop[j:j+opt.batch_size] for prop in sublist]
            flat_dv = [dv for sublist in dv[j:j+opt.batch_size] for dv in sublist]

            encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
            wlen_tensor = torch.FloatTensor(flat_wlen).view(-1, 1)
            prop_tensor = torch.FloatTensor(flat_prop).view(-1, 1)
            dv_tensor = torch.FloatTensor(flat_dv).view(-1, 1)

            encoded_inputs = encoded_inputs.to(opt.device)
            wlen_tensor = wlen_tensor.to(opt.device)
            prop_tensor = prop_tensor.to(opt.device)
            dv_tensor = dv_tensor.to(opt.device)

            optimizer.zero_grad()
            outputs = model(**encoded_inputs, first_idx=first_idx, wlen=wlen_tensor, prop=prop_tensor)
            loss = criterion(outputs, dv_tensor)
            train_loss += loss.item()

            logging.info("Epoch={}, example={}/{}, lr={:3f}, train loss={:3f}, time={:3f}s".format(
                epoch+1, j, len(sent_strings), optimizer.param_groups[0]['lr'], loss.item(), time.time()-start_time))
            loss.backward()

            if opt.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            optimizer.step()
            scheduler.step()
            start_time = time.time()

        if (epoch+1) % opt.evaluate_every == 0:
            model.eval()
            dev_loss = 0
            dev_data_points = 0
            for j in range(0, len(d_sent_strings), opt.dev_batch_size):
                d_batch_sentences = d_sent_strings[j:j+opt.dev_batch_size]
                d_first_idx = d_sent_first_idx[j:j+opt.dev_batch_size]
                d_flat_wlen = [wlen for sublist in d_sent_wlen[j:j+opt.dev_batch_size] for wlen in sublist]
                d_flat_prop = [prop for sublist in d_sent_prop[j:j+opt.dev_batch_size] for prop in sublist]
                d_flat_dv = [dv for sublist in d_dv[j:j+opt.dev_batch_size] for dv in sublist]

                d_encoded_inputs = tokenizer(d_batch_sentences, padding=True, truncation=True, return_tensors="pt")
                d_wlen_tensor = torch.FloatTensor(d_flat_wlen).view(-1, 1)
                d_prop_tensor = torch.FloatTensor(d_flat_prop).view(-1, 1)
                d_dv_tensor = torch.FloatTensor(d_flat_dv).view(-1, 1)

                d_encoded_inputs = d_encoded_inputs.to(opt.device)
                d_wlen_tensor = d_wlen_tensor.to(opt.device)
                d_prop_tensor = d_prop_tensor.to(opt.device)
                d_dv_tensor = d_dv_tensor.to(opt.device)

                outputs = model(**d_encoded_inputs, first_idx=d_first_idx, wlen=d_wlen_tensor, prop=d_prop_tensor)
                loss = criterion(outputs, d_dv_tensor)
                # get total SE by multiplying by number of data points
                dev_loss += loss.item() * len(d_flat_dv)
                dev_data_points += len(d_flat_dv)
                # in_vocab_loss = criterion(outputs[in_vocab_idx], d_dv_tensor[in_vocab_idx])
                # oov_loss = criterion(outputs[oov_idx], d_dv_tensor[oov_idx])

            avg_dev_loss = dev_loss/dev_data_points
            logging.info("=" * 100)
            # logging.info("Epoch={}, train loss={:3f}, dev loss={:3f}, dev in-vocab loss={:3f}, dev oov loss={:3f}, time={:3f}s".format(
            #     epoch + 1, train_loss/math.ceil(len(sent_strings)/opt.batch_size), loss.item(), in_vocab_loss.item(), oov_loss.item(), time.time()-start_time))
            logging.info("Epoch={}, train loss={:3f}, dev loss={:3f}, time={:3f}s".format(epoch+1, train_loss/math.ceil(len(sent_strings)/opt.batch_size),
                                                                                          avg_dev_loss, time.time()-start_time))
            if avg_dev_loss < lowest_mse:
                logging.info("Better model found based on MSE: {}! vs {}".format(avg_dev_loss, lowest_mse))
                lowest_mse = avg_dev_loss
                model_save_path = os.path.join(opt.model_path, "model.pth")
                torch.save(model.state_dict(), model_save_path)
                logging.info("Saved model to {}".format(model_save_path))
            logging.info("=" * 100)
            start_time = time.time()


def test():
    opt = parse_args(sys.argv)
    random_seed(opt.seed, use_cuda=opt.device == "cuda")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logfile = gzip.open(os.path.join(opt.model_path, opt.logfile), "wt")
    writer = SummaryWriter(os.path.join(opt.model_path, "tensorboard"), flush_secs=10)
    filehandler = logging.StreamHandler(logfile)
    streamhandler = logging.StreamHandler(sys.stdout)
    handler_list = [filehandler, streamhandler]
    logging.basicConfig(level="INFO", format="%(asctime)s %(message)s",
                        datefmt="%m/%d/%Y %I:%M:%S %p", handlers=handler_list)
    logging.info(opt)
    writer.add_text("args", str(opt))

    assert (opt.device == "cuda" and torch.cuda.is_available()) or opt.device == "cpu"

    test_data = preprocess.preprocess_test(opt.test_path, opt.pretrained_model)
    logging.info("Test sentences: {}, test tokens: {}.".format(len(test_data[-1]), sum([len(s) for s in test_data[-1]])))

    input_size = {"roberta-base": 768, "roberta-large": 1024}
    roberta = RobertaModel.from_pretrained(opt.pretrained_model)
    tokenizer = RobertaTokenizer.from_pretrained(opt.pretrained_model)
    # model = RobertaForGazePrediction(pretrained=roberta, input_dim=input_size[opt.pretrained_model], dropout_1=opt.dropout_1,
    #                                  hidden_dim=opt.hidden_dim, activation=opt.activation, dropout_2=opt.dropout_2)
    model = RobertaForGazePrediction(pretrained=roberta, input_dim=input_size[opt.pretrained_model],
                                     dropout_1=opt.dropout_1, hidden_dim=int((input_size[opt.pretrained_model]+2)/2), activation=opt.activation,
                                     dropout_2=opt.dropout_2)
    logging.info(str(model))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    logging.info("Topmost model has {} parameters".format(num_params))

    checkpoint = torch.load(opt.checkpoint)
    model.load_state_dict(checkpoint)
    logging.info("Model parameters loaded from {}.".format(opt.checkpoint))

    model = model.to(opt.device)

    sent_strings, sent_first_idx, sent_wlen, sent_prop, sent_words = test_data
    model.eval()
    logging.info("Conducting evaluation on test.")

    all_test_df = []
    for j in range(0, len(sent_strings), opt.dev_batch_size):
        batch_sentences = sent_strings[j:j+opt.dev_batch_size]
        first_idx = sent_first_idx[j:j+opt.dev_batch_size]
        flat_wlen = [wlen for sublist in sent_wlen[j:j+opt.dev_batch_size] for wlen in sublist]
        flat_prop = [prop for sublist in sent_prop[j:j+opt.dev_batch_size] for prop in sublist]
        # flat_word = [word for sublist in sent_words[j:j+opt.dev_batch_size] for word in sublist]

        encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        wlen_tensor = torch.FloatTensor(flat_wlen).view(-1, 1)
        prop_tensor = torch.FloatTensor(flat_prop).view(-1, 1)

        encoded_inputs = encoded_inputs.to(opt.device)
        wlen_tensor = wlen_tensor.to(opt.device)
        prop_tensor = prop_tensor.to(opt.device)

        outputs = model(**encoded_inputs, first_idx=first_idx, wlen=wlen_tensor, prop=prop_tensor)
        logging.info("Generated model predictions on test.")
        outputs_np = outputs.cpu().detach().numpy()
        outputs_df = pd.DataFrame(data=outputs_np, columns=[opt.dep_var])
        # outputs_df["word"] = flat_word
        all_test_df.append(outputs_df)

    merged = pd.concat(all_test_df, ignore_index=True)
    # postprocessing: values below 0.0
    logging.info("The following low values will be replaced with 0.0")
    logging.info(merged.loc[merged[opt.dep_var]<0.0, opt.dep_var])
    merged.loc[merged[opt.dep_var] < 0.0, opt.dep_var] = 0
    # postprocessing: values below 100.0
    logging.info("The following high values will be replaced with 100.0")
    logging.info(merged.loc[merged[opt.dep_var]>100.0, opt.dep_var])
    merged.loc[merged[opt.dep_var]>100.0, opt.dep_var] = 100.0
    merged.to_csv(os.path.join(opt.model_path, "predictions.txt"), sep=" ", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
        logging.shutdown()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        print("Usage: {0} [train|test] [options]".format(sys.argv[0]), file=sys.stderr)
