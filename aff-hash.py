from __future__ import print_function, nested_scopes, unicode_literals, division

from random import choice, shuffle
from scipy.sparse import csc_matrix
# from scipy.spatial.distance import cdist
import numpy as np
# import matplotlib.pyplot as plt
import sys
import timeit
import maxflow

sys.path.append('../liblinear/python')
import liblinearutil

if sys.version_info < (3, 5):
    import cPickle
else:
    import pickle as cPickle

def read_cifar(path='../cifar-10-batches-py/'):
  """ load cifar images
  """
  files = ['data_batch_1',
           'data_batch_2',
           'data_batch_3',
           'data_batch_4',
           'data_batch_5',
           'test_batch',]

  images = []
  labels = []
  for file in files:
    with open(path + file, 'rb') as fo:
      if sys.version_info < (3, 5):
        dict=cPickle.load(fo)
      else:
        dict=cPickle.load(fo, encoding='bytes')
      images.append(dict[b'data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1))
      labels.append(np.asarray(dict[b'labels']).reshape(-1, 1))


  images = np.vstack(images)
  labels = np.vstack(labels).reshape(-1)
  return (images, labels)

def load_gist():
  """ load gist features
  """
  with open('../triplet_hashing-master/features', 'rb') as fo:
    features=cPickle.load(fo)
  return features

def h_step(features, codes, verbose=True):
  N, D = features.shape
  models = []
  for (y, i) in zip(codes.T, range(codes.shape[1])):
    t_start = timeit.default_timer()
    models.append(liblinearutil.train(y.tolist(), features.tolist(), str('-s 0 -c 4 -q')))
    t_end = timeit.default_timer()
    if verbose:
      print('[H] {:3d}th bit, {:.4f} seconds elapsed'.format(i, t_end-t_start))
  return models

def svm_predict(features, models, verbose=True):
  Z = []
  for (m, i) in zip(models, range(len(models))):
    t_start = timeit.default_timer()
    p_label, p_acc, p_val = liblinearutil.predict([0]*features.shape[0], features.tolist(), m , str('-q'))
    Z.append(p_label)
    t_end = timeit.default_timer()
    if verbose:
      print('[P] {:3d}th bit, {:.4f} seconds elapsed'.format(i, t_end-t_start))
  Z = np.vstack(Z).transpose()
  return Z

def generate_pos_neg_idx(labels, num_positive=100, num_negative=500):
  pos=[]
  neg=[]
  for i in range(labels.shape[0]):
    pos_idx = np.nonzero(labels==labels[i])[0]
    pos.append(pos_idx[np.random.choice(pos_idx.shape[0], num_positive)])
    neg_idx = np.nonzero(labels!=labels[i])[0]
    neg.append(neg_idx[np.random.choice(neg_idx.shape[0], num_negative)])
  pos = np.vstack(pos)
  neg = np.vstack(neg)
  # print(pos.shape, neg.shape)
  return (pos, neg)

def loss11(Z, m, n, i, y):
  Zm = Z[m].copy()
  Zn = Z[n].copy()
  Zm[i] = 1
  Zn[i] = 1
  return y * np.linalg.norm(Zm-Zn)**2

def loss01(Z, m, n, i, y):
  Zm = Z[m].copy()
  Zn = Z[n].copy()
  Zm[i] = 1
  Zn[i] = -1
  return y * np.linalg.norm(Zm-Zn)**2

def generate_W(Z, i, pos_idx, neg_idx):
  # W = csc_matrix((labels.shape[0], labels.shape[0]))
  # assert pos_idx.shape[0] == neg_idx.shape[0]
  assert i < Z.shape[1]

  # m_n_l = []
  row = []
  col = []
  l = []
  for m in range(pos_idx.shape[0]):
    for n in pos_idx[m]:
      l.extend([loss11(Z, m, n, i, 1) - loss01(Z, m, n, i, 1)]*2)
      row.extend([m,n])
      col.extend([n,m])
  for m in range(neg_idx.shape[0]):
    for n in neg_idx[m]:
      l.extend([loss11(Z, m, n, i, -1) - loss01(Z, m, n, i, -1)]*2)
      row.extend([m,n])
      col.extend([n,m])

  # print(len(row), len(col), len(l))
  W = csc_matrix((l, (row, col)), shape=(Z.shape[0], Z.shape[0]))

  return W

def generate_submodular(W):
  """Generates a sequence of (possibly overlapping) submodular submatrices of W,
     until all rows have been sampled at least once.

  Args:
      W: matrix with zero diagonal.

  Yields: (sub_W, active_indices): submatrix (sparse) and active indices.

  See Algorithm 1 from Zhuang et al. http://arxiv.org/abs/1603.02844

  """
  num_elements = W.shape[0]
  U = set(range(num_elements))
  while U:
      cur_idx = choice(list(U))
      active_indices = [cur_idx]

      # print ("cur", cur_idx, U)
      U.remove(cur_idx)
      # possible_indices = np.nonzero(W[cur_idx, :].A.ravel() < 0)[0].tolist()
      possible_indices = np.nonzero(W[cur_idx, :].A.ravel() < 0)[0].tolist()
      # print ("poss", possible_indices)
      shuffle(possible_indices)
      for p in possible_indices:
          # if np.all(W[p, :].A.ravel()[active_indices] <= 0):
          if np.all(W[p, :].A.ravel()[active_indices] <= 0):
              active_indices.append(p)
              U.discard(p)
      active_indices = sorted(active_indices)
      # print("act", active_indices)
      # yield W[active_indices, :][:, active_indices].A, active_indices
      yield W[active_indices, :][:, active_indices].A, active_indices

def solve_graphcut(Z, i, h, mu, pos_idx, neg_idx):
  current_labels = np.zeros(Z.shape[0])

  # source = -1, sink = 1
  source_costs = mu * (-np.ones(h.shape[0]) - h[:, i])**2
  sink_costs = mu * (np.ones(h.shape[0]) - h[:, i])**2

  W = generate_W(Z, i, pos_idx, neg_idx)

  for _ in range(2):
    for block, active_indices in generate_submodular(W):
      g = maxflow.GraphFloat()
      g.add_nodes(len(active_indices))
      for (i, idx) in zip(range(len(active_indices)), active_indices):
        g.add_tedge(i, source_costs[idx], sink_costs[idx])

      row, col = np.nonzero(block)
      vals = -block[row, col]
      assert np.all(vals >= 0)
      for i, j, v in zip(row, col, vals):
        g.add_edge(i, j, v, v)
      g.maxflow()

      out = []
      for i in range(len(active_indices)):
        out.append((g.get_segment(i)==1)*2-1)
      current_labels[active_indices] = out

    # print('****')
    # for c in range(10):
    #   ind = np.nonzero(train_labels==c)[0]
    #   print('class {0}, {1} samples,   codes count (1,{2}), \t(-1,{3}).'.format(c,
    #       ind.shape[0],
    #       np.nonzero(current_labels[ind]==1)[0].shape[0],
    #       np.nonzero(current_labels[ind]==-1)[0].shape[0]))

  # print(current_labels)
  return current_labels

def z_step(features, labels, Z, models, mu, num_positive=100, num_negative=500, verbose=True):
  """ Equation (8)
  """
  h = svm_predict(features, models, verbose=verbose)
  # print(h)
  (pos_idx, neg_idx) = generate_pos_neg_idx(labels, num_positive=num_positive, num_negative=num_negative)
  for i in range(Z.shape[1]):
    t_start = timeit.default_timer()
    Z[:, i] = solve_graphcut(Z, i, h, mu, pos_idx, neg_idx)
    t_end = timeit.default_timer()
    if verbose:
      print('[Z] {:3d}th bit, {:.4f} seconds elapsed'.format(i, t_end-t_start))
  return Z


def hash(features, num_train_samples=58000, L=8, verbose=True):
  bits = []
  for i in range(L):
    start = timeit.default_timer()
    m = liblinearutil.load_model('models/tr{0:05d}-L{1:02d}-b{2:02d}.model'.format(num_train_samples, L, i))
    p_label, p_acc, p_val = liblinearutil.predict([0]*features.shape[0], features.tolist(), m , str('-q'))
    bits.append(p_label)
    end = timeit.default_timer()
    if verbose:
      print('[HASH] {0:3d}th bit hashed. {1:.4f} seconds elapsed'.format(i, end-start))

  start = timeit.default_timer()
  bits = np.vstack(bits).transpose().astype(np.int)
  bits[np.nonzero(bits==0)] = -1

  with open('hash/tr{0:05d}-L{1:02d}'.format(num_train_samples, L), 'wb') as fo:
    cPickle.dump(bits, fo)
  end = timeit.default_timer()
  if verbose:
    print('[HASH] Hash codes saved. {0:.4f} seconds elapsed'.format(end-start))
  return

def calc_mean_ap(base_set_labels, num_test, num_train_samples=58000, L=8):
  with open('hash/tr{0:05d}-L{1:02d}'.format(num_train_samples, L), 'rb') as fo:
    codes = cPickle.load(fo)

  assert codes.shape[0]==base_set_labels.shape[0]

  test_labels = base_set_labels[-num_test:]

  distances = -codes[-num_test:].dot(codes.transpose())

  min_idx = np.argsort(distances)
  mean_ap = 0.0
  for i in range(num_test):
    counter = 0
    ap = 0.0
    for j in range(500):
      if base_set_labels[min_idx[i,j]]==test_labels[i]:
        counter = counter + 1
        ap = ap + counter / (j + 1.0)
    if counter == 0:
      counter = 1
    ap = ap / counter
    mean_ap = mean_ap + ap
  mean_ap = mean_ap / num_test

  return mean_ap

def calc_precision_at_k(base_set_labels, num_test, num_train_samples=58000, L=8, K=500):
  with open('hash/tr{0:05d}-L{1:02d}'.format(num_train_samples, L), 'rb') as fo:
    codes = cPickle.load(fo)

  assert codes.shape[0]==base_set_labels.shape[0]

  test_labels = base_set_labels[-num_test:]

  distances = -codes[-num_test:].dot(codes.transpose())

  min_idx = np.argsort(distances)

  p = 0.0
  for i in range(num_test):
    counter = 0
    for j in range(K):
      if base_set_labels[min_idx[i,j]]==test_labels[i]:
        counter = counter + 1
    p = p + counter / (K * 1.0)
  p = p / num_test
  return p

if __name__ == '__main__':
  (color_images, labels) = read_cifar('../data/cifar-10-batches-py/')
  features = load_gist()

  num_train = 10000
  num_test = 2000
  train_features = features[:num_train]
  train_labels = labels[:num_train]
  # test_features = features[-num_test:]

  L = 16
  Z = np.random.randint(2, size=(train_features.shape[0], L)) * 2 - 1
  # mus = [0.001, 0.001, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02]
  mus = [0.001, 0.005]

  for (i, mu) in zip(range(len(mus)), mus):
    print('----------')
    print('[ITER] {:3d} mu = {:.4f}'.format(i + 1, mu))
    t_start = timeit.default_timer()
    models = h_step(train_features, Z, verbose=False)
    t_h = timeit.default_timer()
    print('[H STEP] {0:.4f} seconds elapsed'.format(t_h-t_start))

    Z =z_step(train_features, train_labels, Z, models, mu, 100, 200, verbose=False)
    t_z = timeit.default_timer()
    print('[Z STEP] {0:.4f} seconds elapsed'.format(t_z-t_h))

    for (m,i) in zip(models, range(len(models))):
      liblinearutil.save_model('models/tr{0:05d}-L{1:02d}-b{2:02d}.model'.format(train_features.shape[0], L, i), m)
    t_save = timeit.default_timer()
    print('[SAVE MODEL] {0:.4f} seconds elapsed'.format(t_save-t_z))

    # print('-----------')
    hash(features[:num_train+200], num_train, L, verbose=False)
    t_hash = timeit.default_timer()
    print('[HASH] {0:.4f} seconds elapsed'.format(t_hash-t_save))

    print('Mean average precision: {:.4f}'.format(calc_mean_ap(labels[:num_train+200], 200, num_train, L)))
    print('Precision at K: {:.4f}'.format(calc_precision_at_k(labels[:num_train+200], 200, num_train, L, 20)))
    t_map = timeit.default_timer()
    print('[Precision] {0:.4f} seconds elapsed'.format(t_map-t_hash))

  print('----------')
  hash(features, num_train, L)
  print('final mean average precision: {:.4f}'.format(calc_mean_ap(labels, num_test, num_train, L)))
