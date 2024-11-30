"""
Usage:
    run.py sakt --hidden=<h> --length=<l>[options]

Options:
    --length=<int>                      max length of question sequence [default: 50]
    --questions=<int>                   num of question [default: 124]
    --lr=<float>                        learning rate [default: 0.001]
    --bs=<int>                          batch size [default: 64]
    --seed=<int>                        random seed [default: 59]
    --epochs=<int>                      number of epochs [default: 10]
    --cuda=<int>                        use GPU id [default: 0]
    --hidden=<int>                      dimention of hidden state [default: 128]
    --layers=<int>                      layers of rnn or transformer [default: 1]
    --heads=<int>                       head number of transformer [default: 8]
    --dropout=<float>                   dropout rate [default: 0.1]
"""

import os
import sys
import random
import logging
import torch

import torch.optim as optim
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
# 将需要导入模块代码文件相对于当前文件目录的绝对路径加入到sys.path中
sys.path.append(os.path.join(current_dir, ".."))

from datetime import datetime
from docopt import docopt
from data.dataloader import getDataLoader
from evaluation import eval


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = docopt(__doc__)
    length = int(args['--length'])
    questions = int(args['--questions'])
    lr = float(args['--lr'])
    bs = int(args['--bs'])
    seed = int(args['--seed'])
    epochs = int(args['--epochs'])
    cuda = args['--cuda']
    hidden = int(args['--hidden'])
    layers = int(args['--layers'])
    heads = int(args['--heads'])
    dropout = float(args['--dropout'])


    #--------------日志部分------------------------
    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    date = datetime.now()
    handler = logging.FileHandler(
        f'log/{date.year}_{date.month}_{date.day}_result.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(list(args.items()))

    setup_seed(seed)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    trainLoader, testLoader = getDataLoader(bs, questions, length)

    #选择模型进行运行
    from model.SAKT.model import SAKTModel
    model = SAKTModel(heads, length, hidden, questions, dropout)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = eval.lossFunc(questions, length, device)

    #训练
    for epoch in range(epochs):
        print('epoch: ' + str(epoch))
        model, optimizer = eval.train_epoch(model, trainLoader, optimizer,
                                          loss_func, device)
        logger.info(f'epoch {epoch}')
        pred = eval.test_epoch(model, testLoader, loss_func, device)

    #存放每个学生的知识掌握情况
    knowledge_state = pred[:, -1, :]
    #计算班级的知识掌握情况
    row_means = torch.mean(knowledge_state, dim=0)

    #将学生数据输出到日志中
    for i in range(len(knowledge_state)):
        # 保留小数点后三位
        rounded_values = torch.round(knowledge_state[i] * 1000) / 1000
        logger.info(f"student {i + 1}: {rounded_values}")

    print(row_means)
    #最终输出班级的知识掌握情况
    logger.info("---------------class---------------------")
    logger.info(row_means)

if __name__ == '__main__':
    main()
