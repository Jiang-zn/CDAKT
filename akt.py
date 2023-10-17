import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.cuda.set_device(2)
device = torch.cuda.set_device(0)


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class AKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, q_embed_dim,
                 kq_same, dropout, model_type, final_fc_dim=512, n_heads=8, d_ff=2048, l2=1e-5, separate_qa=False):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            n_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
        """
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = model_type
        self.separate_qa = separate_qa
        self.q_embed_dim = q_embed_dim
        embed_l = d_model
        if self.n_pid > 0:  # 难度嵌入
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question + 1, embed_l)  # 知识点嵌入
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)  # f_(ct,rt)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                  d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff, kq_same=self.kq_same,
                                  model_type=self.model_type)
        # distinct区分度，difficult难度
        self.distout = nn.Sequential(
            nn.Linear(embed_l, 1), nn.ReLU(), nn.Dropout(self.dropout),
        )
        self.diffout = nn.Sequential(
            nn.Linear(embed_l, 1), nn.ReLU(), nn.Dropout(self.dropout),
        )
        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            # nn.Linear(final_fc_dim, q_embed_dim), nn.ReLU(), nn.Dropout(self.dropout),
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def forward(self, q_data, qa_data, target, pid_data=None):
        # Batch First
        q_embed_data = self.q_embed(q_data)  # [BS,seqlen,d_model] c_ct知识点嵌入
        if self.separate_qa:
            qa_embed_data = self.qa_embed(qa_data)  # 嵌入
        else:
            qa_data = torch.div(qa_data - q_data, self.n_question, rounding_mode='floor')  # rt
            # [BS,seqlen,d_model] # c_ct+ g_rt =e_(ct,rt)  概念-响应嵌入
            qa_embed_data = self.qa_embed(qa_data) + q_embed_data

        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 概括了涵盖这个概念的问题的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 控制这个问题与它所涵盖的概念的偏离程度
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data  # c_ct + uq*d_ct，重要公式 Xt
            qa_embed_diff_data = self.qa_embed_diff(qa_data)  # f_(ct,rt) or #h_rt [BS,seqlen,d_model]
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                (qa_embed_diff_data + q_embed_diff_data)  # e_(ct,rt) + uq*(h_rt+d_ct)重要公式 yt
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        d_output = self.model(q_embed_data, qa_embed_data)  # [24,200,256]

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)  # [24,200,512]
        output = self.out(concat_q)  # [24,200,1]，知识掌握情况θ

        dist = self.distout(q_embed_data)  # [24,200,1]
        diff = self.diffout(qa_embed_data)  # [24,200,1]

        # 算每个学生对每一道题的水平，Ability(j)=∑i |Nij| (xji==1) / |Nij|  ,|Nij|>0
        # |Nij|表示学生j作答题目i的次数，xji表示学生j对题目i的回答情况
        # 当学生对每一道题目的水平大于平均水平还答错，失误率si就位学生j错误回答题目i的概率
        # si=∑i |Nij| (xji==0) / |Nij|  ,|Nij|>0
        # 当学生对每一道题目的水平小于平均水平还答对，猜测率gi就为学生j正确回答题目i的概率
        # gi=∑i |Nij| (xji==1) / |Nij|  ,|Nij|>0

        # 计算猜测率g=no_master_right/no_master_total
        no_master_right = torch.empty_like(q_data)  # 没有掌握知识点情况下回答正确的次数no_master_right,对每一个时刻，都计算先前时刻的情况
        no_master_total = torch.empty_like(q_data)  # 没有掌握知识点情况下的总回答次数no_master_total
        no_master_total = torch.cumsum(torch.ones_like(q_data), dim=1)  # no_master_total中dim=1维度的值依次由1到200
        # test_out = torch.sigmoid(output) # output转换为0至1之间的概率值
        # test_out = test_out.squeeze(dim=-1)
        # guessing_rate = 1-test_out
        dist = dist.squeeze(dim=-1)
        diff = diff.squeeze(dim=-1)
        output = output.squeeze(dim=-1)
        correct_answers = (target == 1).float()
        no_master = (output <= 0.5).float()
        no_master_right = torch.cumsum(correct_answers * no_master, dim=1)  # 回答正确的情况*相应时刻未掌握知识点情况，沿时间步累积
        guessing_rate = torch.div(no_master_right, no_master_total)
        guessing_weight = torch.nn.Parameter(torch.tensor(1.0))

        logits = dist*(output)
        # # 预测结果计算公式
        # # P=g+(1-g)*sigmoid(-dist(θ-diff))
        # P = guessing_rate + (1 - guessing_rate) * torch.sigmoid((-dist)*(output - diff))
        # P = torch.sigmoid((-dist)*(output - diff))
        # P = guessing_rate + (1 - guessing_rate) * torch.sigmoid(((-1.7)*dist)*(output - diff))
        # P = torch.sigmoid(((-1.7)*dist)*(output - diff))

        # # P=g+(1-g)*sigmoid(-dist(θ-diff))
        P = (1 - guessing_rate) * torch.sigmoid(logits) + guessing_rate

        labels = target.reshape(-1)
        # m = nn.Sigmoid()
        # preds = (output.reshape(-1))  # logit
        preds = (P.reshape(-1))  # logit
        mask = labels > -0.9
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        return output.sum() + c_reg_loss, preds, mask.sum()


class Architecture(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'akt'}:
            # blocks_1处理qas，blocks_2处理q
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks * 2)
            ])

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False)
                flag_first = False
            else:  # dont peek current response，apply past question sequences and past answer sequences
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same):
        super().__init__()
        """
        This is a Basic Block of Transformer paper. It containts one Multi-head attention object. 
        Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.linear1 = nn.Linear(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)
        Output:
            query: Input gets changed over the layer and returned.
        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')  # 创建一个上三角二维矩阵，mask=0时，遮挡矩阵的对角线及以下部分（模拟padding）
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)  # 遮挡矩阵转换为PyTorch张量是一个二元掩码，在后续的注意力计算中屏蔽掉未来信息
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        # 初始化模型的参数
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    这是由Multi-head atenation对象调用以查找值
    q,k,v是查询，键和值的输入张量.d_k是键的维度，mask是掩码无效位置的张量
    dropout用来正则化，zero_pad指示是否需要零填充，gamma是缩放参数
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    # 这两个矩阵的每个元素代表了序列中的位置信息
    # x1 是一个递增的序列，x2 是 x1 的转置
    # 这两个矩阵被用来计算位置效应的位置矩阵
    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)  # 上三角掩码
        # torch.cumsum(input, dim, dtype=None) 计算输入张量的累积和以及总和
        # 示例
        # x = torch.tensor([1, 2, 3, 4, 5])
        # cumulative_sum = torch.cumsum(x, dim=0)
        # print(cumulative_sum)
        # tensor([ 1,  3,  6, 10, 15])
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
        # 计算 x1 和 x2 之间的差异，距离影响
        # 得到表示位置差异的矩阵 position_effect，用于计算不同位置之间的距离影响
        position_effect = torch.abs(
            x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
        # bs, 8, sl, sl positive distance
        # 计算正数距离dist_scores
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores * gamma).exp(), min=1e-5), max=1e5)
    # 经过距离影响调整后的注意力分数
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
