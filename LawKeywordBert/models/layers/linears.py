import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)
        self.SpanAttention_layer = SpanAttention(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        attn = self.SpanAttention_layer(torch.cat([hidden_states, start_positions], dim=-1),
                                        torch.cat([hidden_states, x], dim=-1))
        x += attn
        return x


class SpanAttention(nn.Module):
    def __init__(self, hidden_size, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, start_position, end_position):
        # end_position_clone = end_position.detach().clone()
        attention_scores = torch.matmul(start_position, end_position.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(end_position.size(-1), dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        x = torch.matmul(attention_weights, end_position)
        x = x + end_position
        x = self.dense_0(x)
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = x + end_position
        x = self.dense_1(x)
        return x


if __name__ == '__main__':
    a = torch.randn((1, 117, 768))
    b = torch.randn((1, 117, 2))

    ep = PoolerEndLogits(768 + 2, 2)
    x = ep(a, b)
    print(x.shape)
