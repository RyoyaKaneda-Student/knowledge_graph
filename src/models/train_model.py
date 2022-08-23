import torch
from torch import nn

from torch import nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################
# Transformer
#######################################

class PositionalEncoder(nn.Module):
    """入力された単語の位置を示すベクトル情報を付加する"""

    # max_seq_len はパディングしないといけない？
    def __init__(self, d_model=768, max_seq_len=100):
        super().__init__()

        self.d_model = d_model  # 単語ベクトルの次元数

        # 単語の順番（pos）と埋め込みベクトルの次元の位置（i）によって一意に定まる値の表をpeとして作成
        pe = torch.zeros(max_seq_len, d_model)
        # print(pe)

        # GPUが使える場合はGPUへ送る
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * (i + 1)) / d_model)))

        # 表peの先頭に、ミニバッチ次元となる次元を足す
        self.pe = pe.unsqueeze(0)

        # 勾配を計算しないようにする
        self.pe.requires_grad = False

    def forward(self, x):

        # 入力xとPositonal Encodingを足し算する
        # xがpeよりも小さいので、大きくする
        ret = math.sqrt(self.d_model) * x + self.pe
        return ret


class Attention(nn.Module):
    """シングルAttentionで実装"""

    def __init__(self, d_model=768):
        super().__init__()

        # SAGANでは1dConvを使用したが、今回は全結合層で特徴量を変換する
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        # 出力時に使用する全結合層
        self.out = nn.Linear(d_model, d_model)

        # Attentionの大きさ調整の変数
        self.d_k = d_model

    def forward(self, cls_tokens, pe):
        # mask = torch.zeros(768)
        # for i in range(0, len(cls_tokens)):

        # 全結合層で特徴量を変換
        k = self.k_linear(pe)
        q = self.q_linear(cls_tokens)
        v = self.v_linear(pe)

        # Attentionの値を計算する
        # 各値を足し算すると大きくなりすぎるので、root(d_k)で割って調整

        # attention_weights = torch.matmul(q, k.transpose(2, 1)) / math.sqrt(self.d_k)

        # attention_weights = torch.matmul(q.unsqueeze(1), k.transpose(2, 1) )/ math.sqrt(self.d_k)
        attention_weights = torch.matmul(q, k.transpose(2, 1)) / math.sqrt(self.d_k)

        # ここでmaskを計算
        # mask = mask.unsqueeze(1)
        # attention_weights = attention_weights.masked_fill(mask == 0, -1e9)

        # softmaxで規格化をする
        normlized_weights = F.softmax(attention_weights, dim=-1)

        # AttentionをValueとかけ算
        output = torch.matmul(normlized_weights, v)

        # 全結合層で特徴量を変換
        output = self.out(output)

        return output, normlized_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, classes, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.classes = classes

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.d_k = d_model

        # 出力時に使用する全結合層
        self.out = nn.Linear(d_model, d_model)
        self._reset_parameters()

        ## def split_heads(self, x, batch_size):
        """最後の次元を(num_heads, depth)に分割。
        結果をshapeが(batch_size, num_heads, seq_len, depth)となるようにリシェイプする。
        """

    ##  x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    ##   return torch.transpose(x, perm=[0, 2, 1, 3])

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.wq.weight)
        self.wq.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.wk.weight)
        self.wk.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.wv.weight)
        self.wv.bias.data.fill_(0)

    def forward(self, cls_tokens, pe):
        # batch_sizeの指定
        batch_size = 4

        q = self.wq(cls_tokens)
        k = self.wk(pe)
        v = self.wv(pe)
        d_k = self.d_k

        q = q.reshape(batch_size, self.classes, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, self.classes, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, self.classes, self.num_heads, self.head_dim)

        # q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        # k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        # v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)

        # attention_weights = torch.matmul(q, k.transpose(-2, -1))
        # attention_weights = attention_weights / math.sqrt(d_k)

        # attention_weights = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1) )/ math.sqrt(self.d_k)
        attention_weights = torch.matmul(q, k.transpose(3, 2)) / math.sqrt(self.d_k)

        # ここでmaskを計算
        # mask = mask.unsqueeze(1)
        # attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
        # softmaxで規格化をする

        normlized_weights = F.softmax(attention_weights, dim=-1)
        # AttentionをValueとかけ算
        values = torch.matmul(normlized_weights, v)
        # 全結合層で特徴量を変換

        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, self.classes, self.d_model)
        output = self.out(values)

        return output, normlized_weights


class FeedForward(nn.Module):
    def __init__(self, d_model=768, d_ff=1536, dropout=0.1):
        '''Attention層から出力を単純に全結合層2つで特徴量を変換するだけのユニットです'''
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model=768, nhead=8, dropout=0.1):
        super().__init__()

        # LayerNormalization層
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # Attention層
        # self.attn = Attention(d_model=d_model)
        self.attn = MultiHeadAttention(d_model=768, classes=7, num_heads=nhead)

        # Attentionのあとの全結合層2つ
        self.ff = FeedForward(d_model)

        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, pn, x):
        # PositionalEncoding を行った後に正規化する
        x_normlized = self.norm_1(x)

        output, normlized_weights = self.attn(pn, x_normlized)
        # FeedForward層の入力作成
        x2 = x + self.dropout_1(output)

        # 正規化と全結合層
        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))

        return output, normlized_weights


class ClassificationHead(nn.Module):
    '''Transformer_Blockの出力を使用し、最後にクラス分類させる'''

    def __init__(self, d_model=768, output_dim=14):
        super().__init__()

        # 全結合層
        self.linear = nn.Linear(d_model, output_dim)  # output_dimはポジ・ネガの2つ

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        # バッチサイズに合わせて変化
        x0 = torch.stack([torch.mean(x[0, :, :], 0), torch.mean(x[1, :, :], 0), torch.mean(x[2, :, :], 0),
                          torch.mean(x[3, :, :], 0)], 0)
        # x0 = x[:, 0, :]  # 各ミニバッチの各文の cls の特徴量（768次元）を取り出す
        out = self.linear(x0)

        # 14 次元に圧縮せずに, 768 次元のマルチラベル分類特化 CLS を獲得
        return out


# 最終的なTransformerモデルのクラス

class TransformerClassification(nn.Module):
    '''Transformerでクラス分類させる'''

    def __init__(self, d_model=768, max_seq_len=100, output_dim=14):
        super().__init__()

        # モデル構築

        # self.net1 = BERT_net
        self.net2 = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.net3_1 = TransformerBlock(d_model=d_model)
        self.net3_2 = TransformerBlock(d_model=d_model)
        self.net4 = ClassificationHead(d_model=d_model, output_dim=output_dim)

    def forward(self, bert_cls1, x1, bert_cls2, x2, bert_cls3, x3, bert_cls4, x4, bert_cls5, x5, bert_cls6, x6,
                bert_cls7, x7):
        q = torch.stack([x1, x2, x3, x4, x5, x6, x7], dim=1)
        k_v = torch.stack([bert_cls1, bert_cls2, bert_cls3, bert_cls4, bert_cls5, bert_cls6, bert_cls7], dim=1)

        # bert_cls, bert_atten, cls_tok,  x1 = self.net1(text)

        # LABEL_CLS, BERT_PRED = LabelCls_convert(bert_cls)

        x13, normlized_weights_1 = self.net3_1(q, k_v)  # Self-Attentionで特徴量を変換
        # x3_1_2 = x13[:, 0, :]

        x3_2, normlized_weights_2 = self.net3_2(q, x13)  # Self-Attentionで特徴量を変換
        tr_cls1 = self.net4(x3_2)  # 768次元を14次元に圧縮

        # OO = nn.Sigmoid()
        return tr_cls1

        # return OO(total_cls)  #normlized_weights_1, #normlized_weights_2

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout = 0.1)
encoder_layer2 = TransformerBlock(d_model=512, nhead=8, dropout = 0.1)

encoder_layer = nn.TransformerEncoderLayer(d_model=200, nhead=8, dropout = 0.1, batch_first=True)
te = nn.TransformerEncoder(encoder_layer, 4)