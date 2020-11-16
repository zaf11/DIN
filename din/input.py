import numpy as np

class DataInput:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    """
    例如某用户点击过 abcd 四 个商品，
    则最终生成的样本为：(其中 X 为随机初始化的某商品 ID) 
    train_set: [(user_id,[a],b,1), (user_id,[a],X1,0), (user_id,[a,b],c,1) 
               ,(user_id,[a,b],X2,0), (user_id,[a,b,c],d,1) (user_id,[a,b,c],X3,0]
    test_set: [(user_id, [a, b, c], (d,X4))]
    """
    u, i, y, sl = [], [], [], []
    for t in ts:
      u.append(t[0]) #user_id
      i.append(t[2]) #item_id
      y.append(t[3]) #label
      sl.append(len(t[1])) #历史行为真实长度，[batch_size]
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int64) #历史行为序列，全部用0填充，[batch_size, max_sl]

    #按照真实的历史行为填充
    k = 0
    for t in ts:
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
      k += 1

    return self.i, (u, i, y, hist_i, sl)#uij


class DataInputTest:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i, j, sl = [], [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[2][0])
      j.append(t[2][1])
      sl.append(len(t[1]))
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int64)

    k = 0
    for t in ts:
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
      k += 1

    return self.i, (u, i, j, hist_i, sl)
