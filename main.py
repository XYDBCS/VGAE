# all the samples are participate training and get the ROC only on miRNA-disease
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import KFold
from utilize import *
import time
import os
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc, precision_score,recall_score, f1_score, precision_recall_curve
from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, mask_test_edges_all_neg, mask_test_edges_local_5FCV

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 700, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 48, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.0 , 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'miRNA-disease', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
#adj, features = load_data(dataset_str)
#print('adj', adj.shape, 'feature', features.shape)



# Step 1. 数据加载

'''
输入为miRNA-miRNA 相似性矩阵M_M， disease-disease相似性矩阵D_D， miRNA-disease邻接矩阵M_D， miRNA-miRNA-disease-disease综合H,
miRNA的数量N_m，disease的数量N_d, miRNA的特征值X_m, disease 的特征值X_d.
'''
M_M = read_txt(".\data\data(383-495)\m-m1.csv")
#M_M = read_txt(".\data\m-m.txt")
N_m = M_M.shape[0]
print('The number of miRNA', N_m)

D_D = read_txt(".\data\data(383-495)\d-d1.csv")
#D_D = read_txt(".\data\d-d.txt")
N_d = D_D.shape[0]
print('The number of disease', N_d)

M_D = read_txt(".\data\data(383-495)\m-d1.csv")
#M_D = read_txt(".\data\m-d.txt")
print(sum(sum(M_D)))
print('The shape of M_D', M_D.shape)

#Step 2 integrate the different data to get the features of each node and the adjacency matrix(including M_M and D-D)
#get the features for miRNA and disease
H =get_feature(M_M, M_D,D_D)
X_m = H[0:N_m, :]
X_d = H[N_m:N_m+N_d, :]
print('The shape of X_m', X_m.shape)

#hetergenerous network
#miRNA-miRNA part, according the different threshold to get the different network
m = np.zeros([N_m, N_m])
#disease-disease connection part according to different threshold to get the different network
def get_new_matrix_MD(D_D, th_d):
    N_d= D_D.shape[0]
    d = np.zeros([N_d, N_d])
    tep_d = 0
    for i in range(N_d):
        for j in range(i):
            if D_D[i][j]>th_d or D_D[j][i]>th_d:
               d[i][j] = 1
               d[j][i]=1
               tep_d = tep_d + 1
            else:
                d[i][j]=0
    return d, tep_d


#step 3, find all the train edges
#adj which is used for getting the samples
d = np.zeros([N_d, N_d])
m = np.zeros([N_m, N_m])
adj111 = np.hstack((m,M_D))
adj222 = np.hstack((M_D.transpose(), d))
adj_MD = np.vstack((adj111,adj222))
label_matrix = adj_MD
#k=47, 56,59,61,85,129,144,187,204,221,270,272,289,321,330
#pos_sample_n = sum(adj_MD[0:577,k+576])
#print("number of samples", pos_sample_n)
adj_MD = sp.coo_matrix(adj_MD)

edges_all, edges_pos, edges_false = mask_test_edges (adj_MD)#将数据分为新的adj矩阵(对称的)，train边（不含对称的），vali边，vali 负边，test边，test负边，
#print(edges_pos.shape)
#np.save('edges_all.npy',edges_all)
#np.save('edges_pos',edges_pos)
#np.save('edges_false',edges_false)
#edges_all = np.load('edges_all.npy')
#edges_pos = np.load('edges_pos.npy')
#edges_false = np.load('edges_false.npy')
X_sample =np.vstack((edges_pos, edges_false)) #all the samples, inlculde all the positive samples and the same number of negative samples
print("edges_pos",len(edges_pos), "edges_neg", len(edges_false) )
print("samples numb in miRNA-disease associations part", len(find_mi_D(X_sample)))
Y_sample = np.hstack((np.ones(len(edges_pos)), np.zeros(len(edges_false))))
###########################################################################################################################################################
# the first section for predicting based on miRNA
###########################################################################################################################################################
th_r1 =0.95
th_d1 =1
m1, tep_N1 = get_new_matrix_MD(M_M, th_r1)
d1, tep_d1 = get_new_matrix_MD(D_D, th_d1)
print("the number of miRNA-MiRNA associations",th_r1,  tep_N1)
print("the number of disease-disease associations",th_d1,  tep_d1)

th_r2 =1
th_d2 = 0.9
m2, tep_N2 = get_new_matrix_MD(M_M, th_r2)
d2, tep_d2 = get_new_matrix_MD(D_D, th_d2)
print("the number of miRNA-MiRNA associations",th_r2,  tep_N2)
print("the number of disease-disease associations",th_d2,  tep_d2)

adj1 = np.hstack((m1,M_D))
adj2 = np.hstack((M_D.transpose(), d1))
adj = np.vstack((adj1,adj2)) # this is the adj matrix after delete some edges

# Store original adjacency matrix (without diagonal entries) for later
adj_orig =sp.coo_matrix(adj)
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape) #adj_orig 变成对角上全为0的矩阵
adj_orig.eliminate_zeros()#将adj_orig变成稀疏存储
adj_or = adj

#step 4 cross validation for training and test samples
labes_all = []
preds_all = []
ite = 0
labes_all_MD = []
preds_all_MD = []
def get_roc_score(edges_pos, edges_neg, pre_matrix, pre_matrix_all, emb=None,):
        if emb is None:
            feed_dict.update({placeholders['dropout']: 0})
            emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        pre_matrix_all = pre_matrix_all + adj_rec
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pre_matrix[e[0],e[1]] = sigmoid(adj_rec[e[0], e[1]])
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            pre_matrix[e[0],e[1]] = sigmoid(adj_rec[e[0], e[1]])
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        return labels_all, preds_all, pre_matrix, pre_matrix_all

kf = KFold(n_splits=5, shuffle=True)

pre_matrix_M = np.zeros([N_d+N_m, N_d+N_m])
pre_matrix_all = np.zeros([N_d+N_m, N_d+N_m])
for train_index, test_index in kf.split(X_sample):
    adj = adj_or
    features = H
    ite = ite +1
    if ite<6:
        train_x_edge = X_sample[train_index, :]
        train_y_label = Y_sample[train_index]
        train_pos = []
        train_neg = []
        train_N = len(train_x_edge)
        # find the positive train samples and negative train samples
        for i in range(train_N):
            if train_y_label [i] == 1:
                train_pos.append(train_x_edge[i,:])
            else:
                train_neg.append(train_x_edge[i,:])
        #find the positive test samples and negative test samples
        test_pos = []
        test_neg = []
        test_x_edge = X_sample[test_index, :]
        test_y_label = Y_sample[test_index]
        test_N = len(test_x_edge)
        for i in range(test_N):
            if test_y_label[i] == 1:
                test_pos.append(test_x_edge[i,:])
            else:
                test_neg.append(test_x_edge[i,:])

        train_pos = np.array(train_pos)
        data = np.ones(train_pos.shape[0])


        # Re-build adj matrix and features
        adj_new = adj
        for i in range(train_pos.shape[0]):
            adj_new[train_pos[i,0], train_pos[i,1]] = 0
            features[train_pos[i,0], train_pos[i,1]] = 0
        adj_all =sp.coo_matrix(adj_new)
        adj_train = adj_orig - sp.dia_matrix((adj_all.diagonal()[np.newaxis, :], [0]), shape=adj_all.shape) #adj_orig 变成对角上全为0的矩阵
        #adj = adj_all.eliminate_zeros()
        adj = adj_train
        features = sp.coo_matrix(H)
       # if FLAGS.features == 0:
        #    features = sp.identity(features.shape[0])  # featureless

        # Some preprocessing
        adj_norm = preprocess_graph(adj)#邻接矩阵归一化处理
        # Define placeholders
        placeholders = { #占位符里面有 特征， 邻接矩阵，原始邻接矩阵， dropout的比例
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }

        num_nodes = adj.shape[0]#网络中结点的个数

        features = sparse_to_tuple(features.tocoo())#特征转化为稀疏矩阵存储,存成tuple的形式，第一行存一对对关系，第二行存每对关系的值，第三行存行列数
        num_features = features[2][1] #特征的维度
        features_nonzero = features[1].shape[0]#有特征的节点个数

        # Create model
        model = None
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()#负样本与正样本的比例
        print('adj.shape', adj.shape[0], adj.sum())
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)# shape[0] = 913, adj.sum() = 10734

        # Optimizer
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Train model
        for epoch in range(FLAGS.epochs):
            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]
        print("Optimization Finished!")

        labes_y, preds_y, pre_matrix_M, pre_matrix_all = get_roc_score(test_pos, test_neg, pre_matrix_M, pre_matrix_all)
        roc_score1 = roc_auc_score(labes_y, preds_y)
        ap_score1 = average_precision_score(labes_y, preds_y)
        print("the training of this time", ite, roc_score1, ap_score1)
        print(ite)
        labes_all.extend(labes_y)
        preds_all.extend(preds_y)
roc_score = roc_auc_score(labes_all, preds_all)
ap_score = average_precision_score(labes_all, preds_all)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
print('prediction result MATRIX ', pre_matrix_M)
#np.save('D22_10_final_predict_matrix_m_heter.npy', pre_matrix_M)

###########################################################################################################################################################
# the second section for predicting based on disease
###########################################################################################################################################################

labes_all = []
preds_all = []
adj11 = np.hstack((m2,M_D))
adj22 = np.hstack((M_D.transpose(), d2))
adjj = np.vstack((adj11,adj22)) # this is the adj matrix after delete some edges

# Store original adjacency matrix (without diagonal entries) for later
adj_orig =sp.coo_matrix(adjj)
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape) #adj_orig 变成对角上全为0的矩阵
adj_orig.eliminate_zeros()#将adj_orig变成稀疏存储
adj_orr = adjj
pre_matrix_D = np.zeros([N_d+N_m, N_d+N_m])
for train_index, test_index in kf.split(X_sample):
    adj = adj_orr
    features = H
    ite = ite +1
    if ite<11:
        train_x_edge = X_sample[train_index, :]
        train_y_label = Y_sample[train_index]
        train_pos = []
        train_neg = []
        train_N = len(train_x_edge)
        # find the positive train samples and negative train samples
        for i in range(train_N):
            if train_y_label [i] == 1:
                train_pos.append(train_x_edge[i,:])
            else:
                train_neg.append(train_x_edge[i,:])
        #find the positive test samples and negative test samples
        test_pos = []
        test_neg = []
        test_x_edge = X_sample[test_index, :]
        test_y_label = Y_sample[test_index]
        test_N = len(test_x_edge)
        for i in range(test_N):
            if test_y_label[i] == 1:
                test_pos.append(test_x_edge[i,:])
            else:
                test_neg.append(test_x_edge[i,:])

        train_pos = np.array(train_pos)
        data = np.ones(train_pos.shape[0])


        # Re-build adj matrix and features
        adj_new = adj
        for i in range(train_pos.shape[0]):
            adj_new[train_pos[i,0], train_pos[i,1]] = 0
            features[train_pos[i,0], train_pos[i,1]] = 0
        adj_all =sp.coo_matrix(adj_new)
        adj_train = adj_orig - sp.dia_matrix((adj_all.diagonal()[np.newaxis, :], [0]), shape=adj_all.shape) #adj_orig 变成对角上全为0的矩阵
        adj = adj_train
        features = sp.coo_matrix(H)
        # Some preprocessing
        adj_norm = preprocess_graph(adj)#邻接矩阵归一化处理
        # Define placeholders
        placeholders = { #占位符里面有 特征， 邻接矩阵，原始邻接矩阵， dropout的比例
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }

        num_nodes = adj.shape[0]#网络中结点的个数

        features = sparse_to_tuple(features.tocoo())#特征转化为稀疏矩阵存储,存成tuple的形式，第一行存一对对关系，第二行存每对关系的值，第三行存行列数
        num_features = features[2][1] #特征的维度
        features_nonzero = features[1].shape[0]#有特征的节点个数

        # Create model
        model = None
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()#负样本与正样本的比例
        print('adj.shape', adj.shape[0], adj.sum())
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)# shape[0] = 913, adj.sum() = 10734

        # Optimizer
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Train model
        for epoch in range(FLAGS.epochs):
            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]
        labes_y, preds_y, pre_matrix_D, pre_matrix_all = get_roc_score(test_pos, test_neg, pre_matrix_D, pre_matrix_all)
        roc_score1 = roc_auc_score(labes_y, preds_y)
        ap_score1 = average_precision_score(labes_y, preds_y)
        print("the training of this time", ite, roc_score1, ap_score1)
        print(ite)
        labes_all.extend(labes_y)
        preds_all.extend(preds_y)
        print('the size of labes_all and preds_all inside', len(labes_all), len(preds_all))
print('the size of labes_all and preds_all', len(labes_all), len(preds_all))
roc_score = roc_auc_score(labes_all, preds_all)
ap_score = average_precision_score(labes_all, preds_all)
print('the shape of labes and preds', len(labes_all), len(preds_all))
print('Test ROC score on only miRNA-disease: ' + str(roc_score))
print('Test AP score on only miRNA-disease: ' + str(ap_score))
print('prediction result MATRIX ', pre_matrix_D)
#np.save('D22_10_final_predict_matrix_heter.npy', pre_matrix_D)

#############################################################################################################################################################################################################################
#getting the final prediction result bsaed on the above two prediction result
#############################################################################################################################################################################################################################
final_predict_matrix = (pre_matrix_D+pre_matrix_M)/2
final_score_matrix = pre_matrix_all/10
np.save('final_score_matrix1.npy', final_score_matrix)
np.save('final_score_matrix1.txt', final_score_matrix)
#np.save('D22_10_Intergrated_final_predict_matrix_heter.npy', final_predict_matrix)
N_all = final_predict_matrix.shape[0]
list_pre = []
list_l = []
for i in range (N_all):
    for j in range(N_all):
        if final_predict_matrix[i,j]>0:
            list_pre.append(final_predict_matrix[i,j])
            list_l.append(label_matrix[i,j])
roc_score_final = roc_auc_score(list_l, list_pre)
ap_score_final = average_precision_score(list_l, list_pre)

print("the final ROC score", roc_score_final)
print("the final AP score", ap_score_final)
print("the len of all the samples", len(list_pre))
