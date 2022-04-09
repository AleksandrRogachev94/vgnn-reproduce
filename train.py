import argparse
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from collections import Counter
import pickle
from tqdm import tqdm
from datetime import datetime
from vgnn_model import VGNN
from utils import train, evaluate, EHRData, collate_fn
import os
import logging
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

# TODO remove. Just for testing
torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser(description='configuraitons')
    parser.add_argument('--result_path', type=str, default='.', help='output path of model checkpoints')
    parser.add_argument('--data_path', type=str, default='./mimc', help='input path of processed dataset')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding size')
    parser.add_argument('--num_of_layers', type=int, default=2, help='number of graph layers')
    parser.add_argument('--num_of_heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
    parser.add_argument('--reg', type=str, default="True", help='regularization')
    parser.add_argument('--lbd', type=float, default=1.0, help='regularization')
    parser.add_argument('--model_path', type=str, default=None, help='Path to checkpoint of trained model')
    parser.add_argument('--none_graph_features', type=int, default=0, help='...')

    args = parser.parse_args()
    result_path = args.result_path
    data_path = args.data_path
    enc_features = args.embedding_size
    dec_features = args.embedding_size
    n_layers = args.num_of_layers
    lr = args.lr
    args.reg = (args.reg == "True")
    n_heads = args.num_of_heads
    dropout = args.dropout
    alpha = 0.1
    BATCH_SIZE = args.batch_size
    model_path = args.model_path
    none_graph_features = args.none_graph_features
    number_of_epochs = 50

    # Load data
    train_x, train_y = pickle.load(open(data_path + 'train_csr.pkl', 'rb'))
    val_x, val_y = pickle.load(open(data_path + 'validation_csr.pkl', 'rb'))
    test_x, test_y = pickle.load(open(data_path + 'test_csr.pkl', 'rb'))
    train_upsampling = np.concatenate((np.arange(len(train_y)), np.repeat(np.where(train_y == 1)[0], 1)))
    train_x = train_x[train_upsampling]
    train_y = train_y[train_upsampling]

    # Create result root
    s = datetime.now().strftime('%Y%m%d%H%M%S')
    result_root = '%s/lr_%s-encoder_%s-decoder_%s-dropout_%s'%(result_path, lr, enc_features, dec_features, dropout)
    if not os.path.exists(result_root):
        os.mkdir(result_root)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info("Time:%s" %(s))

    # initialize models
    device_ids = range(torch.cuda.device_count())
    # eICU has 1 feature on previous readmission that we didn't include in the graph
    model = VGNN(train_x.shape[1], enc_features, dec_features, n_heads, n_layers,
                           dropout=dropout, alpha=alpha, variational=args.reg, none_graph_features=none_graph_features).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)

    # Load existing model, take into account the last epoch
    init_epoch = 0
    if model_path:
        model.load_state_dict(torch.load(model_path))
        init_epoch = int(model_path.split('_')[-1]) + 1
        print('Loaded existing model from: {}'.format(model_path))
        print('Continue training from epoch: {}'.format(init_epoch))

    val_loader = DataLoader(dataset=EHRData(val_x, val_y), batch_size=BATCH_SIZE,
                            collate_fn=collate_fn, num_workers=torch.cuda.device_count(), shuffle=False)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train models
    for epoch in range(init_epoch, number_of_epochs, 1):
        print("Learning rate:{}".format(optimizer.param_groups[0]['lr']))
        ratio = Counter(train_y)
        train_loader = DataLoader(dataset=EHRData(train_x, train_y), batch_size=BATCH_SIZE,
                                  collate_fn=collate_fn, num_workers=torch.cuda.device_count(), shuffle=True)
        pos_weight = torch.ones(1).float().to(device) * (ratio[True] / ratio[False])
        criterion = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)
        t = tqdm(iter(train_loader), leave=False, total=len(train_loader))
        model.train()
        total_loss = np.zeros(3)
        for idx, batch_data in enumerate(t):
            loss, kld, bce = train(batch_data, model, optimizer, criterion, args.lbd, 5)
            total_loss += np.array([loss, bce, kld])
            if idx % 50 == 0 and idx > 0:
                t.set_description('[epoch:%d] loss: %.4f, bce: %.4f, kld: %.4f' %
                                  (epoch + 1, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
                t.refresh()

        torch.save(model.state_dict(), "{}/parameter_epoch_{}".format(result_root, epoch))
        val_auprc, _ = evaluate(model, val_loader, len(val_y))
        logging.info('epoch:%d AUPRC:%f; loss: %.4f, bce: %.4f, kld: %.4f' %
                     (epoch + 1, val_auprc, total_loss[0] / idx, total_loss[1] / idx, total_loss[2] / idx))
        print('epoch:%d AUPRC:%f; loss: %.4f, bce: %.4f, kld: %.4f' %
              (epoch + 1, val_auprc, total_loss[0] / idx, total_loss[1] / idx, total_loss[2] / idx))

        scheduler.step()


if __name__ == '__main__':
    main()