# coding=utf-8
#%% 导入包
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import os
from sklearn.model_selection import train_test_split
import time

# 1. 准备数据 ===========================================================================
# %% 处理并加载数据，加上开始和结束标记
def preprocess_sentence(w):
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    return w

# 加载数据
def load_data(num = 10000):

    # 保存上联（input）和下联(target)的数组
    inp_lang = []
    targ_lang = []

    # 指定文件位置（同级目录data下的data.txt文件）
    file = open('C://all.txt','rb')
    line = file.readline()
    # 读取一行文本，如果此行存在
    while line: 
        # 读出内容：【春 来 眼 际 ， 喜 上 眉 梢】拆分上下联
        wstr = str(line, encoding = "utf-8")
        wstrs = wstr.split('，')
        if len(wstrs) > 1:
            inp_lang.append(preprocess_sentence(wstrs[0]))
            targ_lang.append(preprocess_sentence(wstrs[1]))
        # 接着再读下一行
        line = file.readline()
        #　如果超过预设最大数量，则退出循环
        if len(inp_lang) > num:break 
    file.close()

    return inp_lang, targ_lang

inp_lang, targ_lang = load_data()
# print("inp_lang=>","\n ",inp_lang,"inp_lang=>","\n ", targ_lang)
'''
inp_lang=> ['<start> 春 来 眼 际 <end>', '<start> 春 光 普 照 <end>', '<start> 春 和 景 明 <end>'] 
inp_lang=> ['<start> 喜 上 眉 梢 <end>', '<start> 福 气 长 临 <end>', '<start> 物 阜 年 丰 <end>']
'''

#%%
# 分词器，将文本转化为序列：上天言好事->65234
def tokenize(lang):
    # 创建分词器，默认以空格分词
    lang_tokenizer = Tokenizer(oov_token='<OOV>', filters='',split=' ')
    # 分词器设置处理训练的文本。  {'<end>': 2,'<start>': 1,'下': 54,'不': 5}
    lang_tokenizer.fit_on_texts(lang) 
    #print("lang_tokenizer.index_word=>","\n ", lang_tokenizer.index_word)

    # 分词器序列化文本为数字。[[18, 19],[1, 18, 19,20]]
    tensor = lang_tokenizer.texts_to_sequences(lang)
    #print("tensor=>","\n ", tensor)

    max_sequence_len = max([len(x) for x in tensor])
    #print("max_sequence_len=>","\n ", max_sequence_len)
    #print("lang_tokenizer.word_index=>","\n ", len(lang_tokenizer.word_index))
    # 预处理文本，补齐长度，统一格式 [[0, 0, 18, 19],[1, 18, 19, 20]]
    tensor = pad_sequences(tensor, maxlen=max_sequence_len)
    #print("pad_sequences tensor=>","\n ", tensor)

    # 将序列化后的数字和包含段落信息的分词器返回
    return tensor, lang_tokenizer

input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

# 包装成TF需要的格式，采用8:2切分训练集和验证集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
BUFFER_SIZE = len(input_tensor_train) # 训练集的大小
BATCH_SIZE = 64 # 批次大小，一次取多少条数据
# 一轮训练，需要几步取完所有数据
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE  
# 将数据集顺序打乱
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
# 生成训练批次，N批64条的数据drop_remainder=True最后一个批次不够的舍弃，去除余数。
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# %%
# 2. 组建编码器和解码器 ===========================================================================
embedding_dim = 256 # 维度
units = 1024 # 神经元数量
# 输入集字库的数量，+1是因为有长度不够时，为了凑数的补0
vocab_inp_size = len(inp_lang_tokenizer.word_index)+1 
# 输出集字库的数量，+1是因为有0
vocab_tar_size = len(targ_lang_tokenizer.word_index)+1  

# 编码器，把数据转化为向量
class Encoder(tf.keras.Model):

  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz #　每次批次大小
    self.enc_units = enc_units # 神经元数量
    # 嵌入层（字符数量，维度数）
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    # 门控循环单元 GRU
    self.gru = tf.keras.layers.GRU(self.enc_units, #　神经元数量
                                   return_sequences=True, # 返回序列
                                   return_state=True, # 返回状态
                                   recurrent_initializer='glorot_uniform') # 选择初始化器，固定值

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# 注意力机器
class BahdanauAttention(tf.keras.layers.Layer):
    
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # 隐藏层的形状 == （批大小，隐藏层大小）
        # hidden_with_time_axis 的形状 == （批大小，1，隐藏层大小）
        # 这样做是为了执行加法以计算分数  
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # 分数的形状 == （批大小，最大长度，1）
        # 我们在最后一个轴上得到 1， 因为我们把分数应用于 self.V
        # 在应用 self.V 之前，张量的形状是（批大小，最大长度，单位）
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # 注意力权重 （attention_weights） 的形状 == （批大小，最大长度，1）
        attention_weights = tf.nn.softmax(score, axis=1)

        # 上下文向量 （context_vector） 求和之后的形状 == （批大小，隐藏层大小）
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# 解码器
class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # 用于注意力
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # 编码器输出 （enc_output） 的形状 == （批大小，最大长度，隐藏层大小）
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        x = self.embedding(x)

        # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 将合并后的向量传送到 GRU
        output, state = self.gru(x)

        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(output, (-1, output.shape[2]))

        # 输出的形状 == （批大小，vocab）
        x = self.fc(output)

        return x, state, attention_weights

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

# 3. 训练数据 ===========================================================================
# %% 损失函数
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

checkpoint_dir = 'F://books/tf_test/juejin_chunlian/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden):
    
    loss = 0

    with tf.GradientTape() as tape:

        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
        # 强制 - 将目标词作为下一个输入
        for t in range(1, targ.shape[1]):
            # 将编码器输出 （enc_output） 传送至解码器
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            # 使用强制
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

EPOCHS = 300

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 10 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch,batch_loss.numpy()))
    # 每 2 个周期（epoch），保存（检查点）一次模型
    if (epoch + 1) % 10 == 0:
        print('save model')
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# 4. 验证数据 ===========================================================================
# %%
def max_length(tensor):
    return max(len(t) for t in tensor)
# 计算目标张量的最大长度 （max_length）
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

def evaluate(sentence):

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang_tokenizer.word_index[i] for i in sentence.split(' ')]
    print("inputs:",inputs)
    inputs = pad_sequences([inputs], maxlen=max_length_inp)

    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']], 0)
    outputs = []
    for t in range(max_length_targ):

        predictions, dec_hidden, attention_weights = decoder(dec_input,dec_hidden, enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()
        outputs.append(predicted_id)
        result += targ_lang_tokenizer.index_word[predicted_id] + ' '

        if targ_lang_tokenizer.index_word[predicted_id] == '<end>':
            print("outputs:",outputs)
            return result, sentence

        # 预测的 ID 被输送回模型
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence

# 恢复检查点目录 （checkpoint_dir） 中最新的检查点
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# %%
s = '百 花 争 艳 春 风 得 意'
# l = list(s)
# s = " ".join(s)
result, sentence = evaluate(s)
print('Input: %s' % (sentence))
print('Predicted: {}'.format(result))
# %%
