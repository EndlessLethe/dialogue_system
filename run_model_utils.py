'''
Author: Zeng Siwei
Date: 2020-08-27 23:28:51
LastEditors: Zeng Siwei
LastEditTime: 2020-08-28 10:15:10
Description: 
'''
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime

def run_model(model, loss_fn, optimizer, dataloader_train, dataloader_dev, dataloader_test, batch_size, use_gpu, patience_max = 5):
    logging.info("creating model and loading data...")
    model_name = str(model)
    logging.info(model_name)

    if use_gpu:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    best_loss = 10000
    patience_count = 0
    for epoch in range(100):
        logging.info("="*48)
        logging.info("epoch " + str(epoch+1))
        logging.info("="*48)

        train_epoch(model, loss_fn, optimizer, dataloader_train, use_gpu)
        eval_loss = eval_epoch(model, loss_fn, optimizer, dataloader_dev, use_gpu)

        if eval_loss < best_loss:
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience_max:
                print('-> Early stopping at epoch {}...'.format(epoch))
                break


    x_label, x_pred = predict(model, dataloader_test, use_gpu)
    logging.info(x_label)
    data_pred = pd.DataFrame(x_pred)
    data_pred.to_csv("./output/" + model_name + str(datetime.now()) + ".csv", sep="\t", header = None, index = False)


def train_epoch(model, loss_fn, optimizer, dataloader, batch_size, use_gpu):
    model.train()

    loss_train = 0
    cnt_acc = 0
    for i, data in enumerate(dataloader):
        # 将数据从 train_loader 中读出来,一次读取的样本数是batch_size个

        inputs, labels = data

        if use_gpu:
            inputs = [x.cuda() for x in inputs]
            labels = labels.cuda()

        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        loss_train += loss.item()
        y_label = torch.argmax(y_pred, dim = 1)
        cnt_acc += torch.sum(labels.data == y_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 199:
            logging.info('step %5d loss_train: %.3f' % (i + 1, loss_train / 200.0))
            logging.info('step %5d acc_train: %.3f' % (i + 1, cnt_acc / 200.0 / batch_size))
            loss_train = 0.0
            cnt_acc = 0

def eval_epoch(model, loss_fn, optimizer, dataloader, batch_size, use_gpu):
    model.eval()

    with torch.no_grad():
        loss_eval = 0
        cnt_acc = 0
        total_loss = 0
        total_cnt = 0
        for i, data in enumerate(dataloader):
            # 将数据从 train_loader 中读出来,一次读取的样本数是batch_size个

            inputs, labels = data

            if use_gpu:
                inputs = [x.cuda() for x in inputs]
                labels = labels.cuda()

            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            loss_eval += loss.item()
            total_loss += loss.item()

            y_label = torch.argmax(y_pred, dim = 1)
            cnt_acc += torch.sum(labels.data == y_label)
            total_cnt += labels.size()[0]

            if i % 20 == 19:
                logging.info('step %5d loss_eval: %.3f' % (i + 1, loss_eval / 20.0))
                logging.info('step %5d acc_eval: %.3f' % (i + 1, cnt_acc / 20.0 / batch_size))

                loss_eval = 0.0
                cnt_acc = 0

    return total_loss/total_cnt


def predict(model, dataloader, use_gpu):
    model.eval()

    list_pred = []
    list_label = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # 将数据从 train_loader 中读出来,一次读取的样本数是batch_size个
            inputs, labels = data

            if use_gpu:
                inputs = [x.cuda() for x in inputs]
                labels = labels.cuda()

            y_pred = model(inputs).cpu()

            for pred in y_pred:
                label = torch.argmax(pred)
                list_label.append(label.numpy())
                list_pred.append(torch.nn.functional.softmax(pred).numpy())

    return np.array(list_label), np.array(list_pred)