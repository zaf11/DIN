import tensorflow as tf
from Dice import dice


class Model(object):
    """
    user_count: 192403, item_count: 63001, cate_count: 801, example_count: 1689188
    cate_list:每个商品的类别，len(cate_list)=63001
    """
    def __init__(self, user_count, item_count, cate_count, cate_list):
        # shape: [B],  user id。 (B：batch size)
        self.u = tf.placeholder(tf.int32, [None, ])
        # shape: [B]  i: 正样本的item
        self.i = tf.placeholder(tf.int32, [None, ])
        # shape: [B]  j: 负样本的item
        self.j = tf.placeholder(tf.int32, [None, ])
        # shape: [B], y: label
        self.y = tf.placeholder(tf.float32, [None, ])
        # shape: [B, T] #用户行为特征(User Behavior)中的item序列。
        # T为最长序列的长度，用户序列少于T的用0填充
        self.hist_i = tf.placeholder(tf.int32, [None, None])
        # shape: [B]; sl：sequence length，User Behavior中每个用户行为序列的真实长度
        self.sl = tf.placeholder(tf.int32, [None, ])
        # learning rate
        self.lr = tf.placeholder(tf.float64, [])

        hidden_units = 128
        # shape: [U, H], user_id的embedding weight. U是user_id的hash bucket size
        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
        # shape: [I, H//2], item_id的embedding weight. I是item_id的hash bucket size
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])  # [I, H//2]
        # shape: [I], item_id的embedding bias
        item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))
        # shape: [C, H//2], cate_id的embedding weight.
        cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
        # shape: [C, H//2]
        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

        # 正样本的cate
        ic = tf.gather(cate_list, self.i)
        # 正样本item_emb和cate_emb拼接，shape: [B, H]
        i_emb = tf.concat(values=[tf.nn.embedding_lookup(item_emb_w, self.i), tf.nn.embedding_lookup(cate_emb_w, ic), ], axis=1)
        # 偏置b
        i_b = tf.gather(item_b, self.i)

        # 从cate_list中取出负样本的cate
        jc = tf.gather(cate_list, self.j)
        # 负样本item_emb和cate_emb拼接，shape: [B, H]
        j_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.j), tf.nn.embedding_lookup(cate_emb_w, jc), ], axis=1)
        # 偏置b
        j_b = tf.gather(item_b, self.j)

        # 用户行为序列(User Behavior)中的cate序列
        hc = tf.gather(cate_list, self.hist_i)

        # 用户行为序列(User Behavior)物品的item_emb和cate_emb拼接，shape: [B, T, H]
        h_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.hist_i), tf.nn.embedding_lookup(cate_emb_w, hc), ],
                          axis=2)
        # attention操作
        hist_i = attention(i_emb, h_emb, self.sl) # B * 1 * H
        # -- attention end ---

        hist = tf.layers.batch_normalization(inputs=hist_i)
        hist = tf.reshape(hist, [-1, hidden_units])# B * H

        # 添加一层全连接层，hist为输入，hidden_units为输出维数
        hist = tf.layers.dense(hist, hidden_units) #为啥？

        u_emb = hist

        # 下面两个全连接用来计算y'，i为正样本，j为负样本
        # fcn begin
        din_i = tf.concat([u_emb, i_emb], axis=-1)
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=None, name='f1')
        d_layer_1_i = dice(d_layer_1_i, name='dice_1_i')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=None, name='f2')
        d_layer_2_i = dice(d_layer_2_i, name='dice_2_i')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')

        din_j = tf.concat([u_emb, j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=None, name='f1', reuse=True)
        d_layer_1_j = dice(d_layer_1_j, name='dice_1_j')
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=None, name='f2', reuse=True)
        d_layer_2_j = dice(d_layer_2_j, name='dice_2_j')
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)

        d_layer_3_i = tf.reshape(d_layer_3_i, [-1]) # [B, 1]
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1]) # [B, 1]

        # 预测的（y正-y负）
        x = i_b - j_b + d_layer_3_i - d_layer_3_j  # [B]
        # 预测的（y正）
        self.logits = i_b + d_layer_3_i

        # logits for all item:
        u_emb_all = tf.expand_dims(u_emb, 1)
        u_emb_all = tf.tile(u_emb_all, [1, item_count, 1])
        # 将所有的除u_emb_all外的embedding，concat到一起
        all_emb = tf.concat([item_emb_w, tf.nn.embedding_lookup(cate_emb_w, cate_list)], axis=1)
        all_emb = tf.expand_dims(all_emb, 0)
        all_emb = tf.tile(all_emb, [512, 1, 1])
        # 将所有的embedding，concat到一起
        din_all = tf.concat([u_emb_all, all_emb], axis=-1)
        din_all = tf.layers.batch_normalization(inputs=din_all, name='b1', reuse=True)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=None, name='f1', reuse=True)
        d_layer_1_all = dice(d_layer_1_all, name='dice_1_all')
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=None, name='f2', reuse=True)
        d_layer_2_all = dice(d_layer_2_all, name='dice_2_all')
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3', reuse=True)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, item_count])

        self.logits_all = tf.sigmoid(item_b + d_layer_3_all)
        # -- fcn end -------

        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        self.score_i = tf.sigmoid(i_b + d_layer_3_i)
        self.score_j = tf.sigmoid(j_b + d_layer_3_j)
        self.score_i = tf.reshape(self.score_i, [-1, 1])
        self.score_j = tf.reshape(self.score_j, [-1, 1])
        self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        # loss and train
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        trainable_params = tf.trainable_variables()
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)

    #uij:(u, i, y, hist_i, sl)
    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            # self.u : uij[0],
            self.i: uij[1], #user_id
            self.y: uij[2], #item_id
            self.hist_i: uij[3], #[batch_size, T],T为batch_size中最长的历史行为序列长度，不满足T的行数据用0填充
            self.sl: uij[4], #历史行为真实长度，[batch_size]
            self.lr: lr
        })
        return loss

    def eval(self, sess, uij):
        u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
            # self.u: uij[0],
            self.i: uij[1],  # 正样本
            self.j: uij[2],  # 负样本
            self.hist_i: uij[3],
            self.sl: uij[4],
        })
        return u_auc, socre_p_and_n

    def test(self, sess, uid, hist_i, sl):
        return sess.run(self.logits_all, feed_dict={
            self.u: uid,
            self.hist_i: hist_i,
            self.sl: sl,
        })

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


# item_embedding,history_behivior_embedding,sequence_length
def attention(queries, keys, keys_length):
    '''
        queries:     [B, H]    [batch_size,embedding_size]
        keys:        [B, T, H]   [batch_size,T,embedding_size]
        keys_length: [B]        [batch_size]
        #T为历史行为序列长度
    '''

    # tile()函数是用来对张量(Tensor)进行扩展的，其特点是对当前张量内的数据进行一定规则的复制。最终的输出张量维度不变
    # 对queries的维度进行reshape
    # (?,T,32)这里是为了让queries和keys的维度相同而做的操作
    # (?,T,128)把u和v以及u v的element wise差值向量合并起来作为输入，
    # 然后喂给全连接层，最后得出两个item embedding，比如u和v的权重，即g(Vi,Va)

    queries_hidden_units = queries.get_shape().as_list()[-1]  #queries_hidden_units = H
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])  #tf.shape(keys)[1] = T
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])  #[B, T, H]
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # B*T*4H

    # 三层全链接(d_layer_3_all为训练出来的atteneion权重）
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')  # B*T*1

    # 为了让outputs维度和keys的维度一致
    outputs = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # B*1*T

    #tf.shape(keys)[1]==T
    #  tf.sequence_mask([1, 2, 3]，3)
    #  即为[[ True False False]
    #       [ True  True False]
    #       [ True  True  True]]
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  #[B*T]

    # 在第二维增加一维，也就是由B*T变成B*1*T
    key_masks = tf.expand_dims(key_masks, 1)  # B*1*T

    # tf.ones_like新建一个与output类型大小一致的tensor，设置填充值为一个很小的值，而不是0,padding的mask后补一个很小的负数，这样softmax之后就会接近0
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

    # 填充，获取等长的行为序列
    # tf.where(condition， x, y),condition是bool型值，True/False，返回值是对应元素，condition中元素为True的元素替换为x中的元素，为False的元素替换为y中对应元素
    # 由于是替换，返回值的维度，和condition，x ， y都是相等的。
    outputs = tf.where(key_masks, outputs, paddings)  # B * 1 * T

    # Scale（缩放）
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
    # Activation
    outputs = tf.nn.softmax(outputs)  # B * 1 * T
    # Weighted Sum outputs=g(Vi,Va)   keys=Vi
    # 这步为公式中的g(Vi*Va)*Vi
    outputs = tf.matmul(outputs, keys)  # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))

    return outputs