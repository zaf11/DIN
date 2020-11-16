import random
import pickle

random.seed(1234)

with open('../raw_data/remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count, example_count = pickle.load(f)

print("cate_list:", cate_list)
train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
  pos_list = hist['asin'].tolist()
  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      neg = random.randint(0, item_count-1)
    return neg
  #与正样本等长
  neg_list = [gen_neg() for i in range(len(pos_list))]

  ''' 
  点击数需要大于1，否则舍弃用户数据
  下面的 if 语句控制正负样本的个数和格式），例如某用户点击过 abcd 四 个商品，
  则最终生成的样本为：(其中 X 为随机初始化的某商品 ID) 
  train_set: [(user_id,[a],b,1), (user_id,[a],X1,0), (user_id,[a,b],c,1) 
             ,(user_id,[a,b],X2,0), (user_id,[a,b,c],d,1) (user_id,[a,b,c],X3,0]
  test_set: [(user_id, [a, b, c], (d,X4))]
  '''
  if reviewerID == 0:
    print("pos_list:", pos_list)
  for i in range(1, len(pos_list)):
    hist = pos_list[:i]
    if i != len(pos_list) - 1:
      train_set.append((reviewerID, hist, pos_list[i], 1))
      train_set.append((reviewerID, hist, neg_list[i], 0))
    else:
      label = (pos_list[i], neg_list[i])
      test_set.append((reviewerID, hist, label))

print("train_set:",train_set[:10])
print("test_set:",test_set[:10])

random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

with open('dataset.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
