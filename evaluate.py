import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from vgnn_model import VGNN
from utils import evaluate, EHRData, collate_fn
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

# TODO remove. Just for testing
torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser(description='configurations')
    parser.add_argument('--data_path', type=str, default='./mimc', help='input path of processed dataset')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding size')
    parser.add_argument('--num_of_layers', type=int, default=2, help='number of graph layers')
    parser.add_argument('--num_of_heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
    parser.add_argument('--reg', type=str, default="True", help='regularization')
    parser.add_argument('--lbd', type=int, default=1.0, help='regularization')
    parser.add_argument('--model_path', type=str, default=None, help='Path to checkpoint of trained model')
    parser.add_argument('--none_graph_features', type=int, default=0, help='...')

    args = parser.parse_args()
    data_path = args.data_path
    enc_features = args.embedding_size
    dec_features = args.embedding_size
    n_layers = args.num_of_layers
    reg = (args.reg == "True")
    n_heads = args.num_of_heads
    dropout = args.dropout
    alpha = 0.1
    BATCH_SIZE = args.batch_size
    model_path = args.model_path
    none_graph_features = args.none_graph_features

    # Load data
    test_x, test_y = pickle.load(open(data_path + 'test_csr.pkl', 'rb'))

    # initialize models
    device_ids = range(torch.cuda.device_count())
    # eICU has 1 feature on previous readmission that we didn't include in the graph
    model = VGNN(test_x.shape[1], enc_features, dec_features, n_heads, n_layers,
                           dropout=dropout, alpha=alpha, variational=reg, none_graph_features=none_graph_features).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)

    # Load existing model, take into account the last epoch
    model.load_state_dict(torch.load(model_path))
    print('Loaded existing model from: {}'.format(model_path))

    test_loader = DataLoader(dataset=EHRData(test_x, test_y), batch_size=BATCH_SIZE,
                            collate_fn=collate_fn, num_workers=torch.cuda.device_count(), shuffle=False)

    test_auprc, _ = evaluate(model, test_loader, len(test_y))
    print('AUPRC: %f' % (test_auprc))


if __name__ == '__main__':
    main()