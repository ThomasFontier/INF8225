'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import tensorflow as tf
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../..')))
from DGFraud.algorithms.SemiGNN.SemiGNN import SemiGNN
import time
from utils.data_loader import *
from utils.utils import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def process_data():
    data = pd.read_csv(r'../../../S-FFSD.csv')
    data = data[data['Labels'] != 2]
    print(data.head())

    X = data.drop(['Labels'], axis=1)
    y = data['Labels']

    # on a décidé de faire de l'oversampling sur les fraudes suivi d'un undersampling sur les non-fraudes
    # l'idée est d'équilibrer les deux classdes sans pour autant avoir à créer trop de données de fraudes
    over = RandomOverSampler(sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=1.0)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    encoder = LabelEncoder()
    combined_nodes = pd.concat([X_resampled['Source'], X_resampled['Target']])
    encoder.fit(combined_nodes)

    X_resampled['Source'] = encoder.transform(X_resampled['Source'])
    X_resampled['Target'] = encoder.transform(X_resampled['Target'])

    features = pd.get_dummies(X_resampled[['Source', 'Target', 'Location', 'Type']])
    
    N = encoder.classes_.size 
    adjacency_matrix = np.zeros((N, N))
    weighted_adjacency_matrix = np.zeros((N, N))

    for _, row in X_resampled.iterrows():
        adjacency_matrix[row['Source'], row['Target']] = 1
        weighted_adjacency_matrix[row['Source'], row['Target']] += row['Amount']

    X_train, X_test, y_train, y_test = train_test_split(range(len(y_resampled)), np.eye(2)[y_resampled], test_size=0.2, random_state=48, stratify=y_resampled)
    
    return [adjacency_matrix, weighted_adjacency_matrix], features, X_train, y_train, X_test, y_test

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# init the common args, expect the model specific args
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--dataset_str', type=str, default='example', help="['dblp','example']")
    parser.add_argument('--epoch_num', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--momentum', type=int, default=0.9)
    parser.add_argument('--learning_rate', default=0.001, help='the ratio of training set in whole dataset.')

    # SemiGNN
    parser.add_argument('--init_emb_size', default=4, help='initial node embedding size')
    parser.add_argument('--semi_encoding1', default=3, help='the first view attention layer unit number')
    parser.add_argument('--semi_encoding2', default=2, help='the second view attention layer unit number')
    parser.add_argument('--semi_encoding3', default=4, help='one-layer perceptron units')
    parser.add_argument('--Ul', default=8, help='labeled users number')
    parser.add_argument('--alpha', default=0.5, help='loss alpha')
    parser.add_argument('--lamtha', default=0.5, help='loss lamtha')

    args = parser.parse_args()
    return args


def set_env(args):
    tf.reset_default_graph()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)


# get batch data
def get_data(ix, int_batch, train_size):
    if ix + int_batch >= train_size:
        ix = train_size - int_batch
        end = train_size
    else:
        end = ix + int_batch
    return train_data[ix:end], train_label[ix:end]


def load_data(args):
    if args.dataset_str == 'example':
        adj_list, features, train_data, train_label, test_data, test_label = process_data()
        node_size = features.shape[0]
        node_embedding = features.shape[1]
        class_size = train_label.shape[1]
        train_size = len(train_data)
        paras = [node_size, node_embedding, class_size, train_size]

    return adj_list, features, train_data, train_label, test_data, test_label, paras


def train(args, adj_list, features, train_data, train_label, test_data, test_label, paras):
    with tf.Session() as sess:
        adj_nodelists = [matrix_to_adjlist(adj, pad=False) for adj in adj_list]
        meta_size = len(adj_list)
        pairs = [random_walks(adj_nodelists[i], 2, 3) for i in range(meta_size)]
        net = SemiGNN(session=sess, class_size=paras[2], semi_encoding1=args.semi_encoding1,
                      semi_encoding2=args.semi_encoding2, semi_encoding3=args.semi_encoding3,
                      meta=meta_size, nodes=paras[0], init_emb_size=args.init_emb_size, ul=args.batch_size,
                      alpha=args.alpha, lamtha=args.lamtha)
        adj_data = [pairs_to_matrix(p, paras[0]) for p in pairs]
        u_i = []
        u_j = []
        for adj_nodelist, p in zip(adj_nodelists, pairs):
            u_i_t, u_j_t, graph_label = get_negative_sampling(p, adj_nodelist)
            u_i.append(u_i_t)
            u_j.append(u_j_t)
        u_i = np.concatenate(np.array(u_i))
        u_j = np.concatenate(np.array(u_j))

        sess.run(tf.global_variables_initializer())
        # net.load(sess)

        t_start = time.clock()
        for epoch in range(args.epoch_num):
            train_loss = 0
            train_acc = 0
            count = 0
            for index in range(0, paras[3], args.batch_size):
                batch_data, batch_sup_label = get_data(index, args.batch_size, paras[3])
                loss, acc, pred, prob = net.train(adj_data, u_i, u_j, graph_label, batch_data,
                                                  batch_sup_label,
                                                  args.learning_rate,
                                                  args.momentum)

                print("batch loss: {:.4f}, batch acc: {:.4f}".format(loss, acc))
                # print(prob, pred)

                train_loss += loss
                train_acc += acc
                count += 1
            train_loss = train_loss / count
            train_acc = train_acc / count
            print("epoch{:d} : train_loss: {:.4f}, train_acc: {:.4f}".format(epoch, train_loss, train_acc))
            # net.save(sess)

        t_end = time.clock()
        print("train time=", "{:.5f}".format(t_end - t_start))
        print("Train end!")

        test_acc, test_pred, test_probabilities, test_tags = net.test(adj_data, u_i, u_j,
                                                                      graph_label,
                                                                      test_data,
                                                                      test_label,
                                                                      args.learning_rate,
                                                                      args.momentum)

    print("test acc:", test_acc)

if __name__ == "__main__":
    args = arg_parser()
    set_env(args)
    adj_list, features, train_data, train_label, test_data, test_label, paras = load_data(args)
    train(args, adj_list, features, train_data, train_label, test_data, test_label, paras)
