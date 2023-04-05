import torch
import torch.nn as nn
import torch.nn.init as init


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_ctx, config):
        super(MaskedMultiHeadAttention, self).__init__()
        self.n_head = config.n_head
        assert hidden_size % self.n_head == 0

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # (1, 1, 1024, 1024)
        self.masked = torch.tril(torch.ones(n_ctx, n_ctx).view(1, 1, n_ctx, n_ctx))
        self.masked_bias = torch.tensor(-1e4)

    def split_heads(self, x, k=False):
        # size is (batch_size, seq_len, num_head, head_features)
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)

        # key is transpose
        if k:
            # (batch_size, num_head, head_features, seq_len)
            return x.permute(0, 2, 3, 1)
        else:
            # (batch_size, num_head, seq_len, head_features)
            return x.permute(0, 2, 1, 3)

    @staticmethod
    def merge_heads(x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def forward(self, hidden_states, layer_past=None, attention_mask=None):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            # (batch_size, num_head, head_features, sql_len+1)
            key = torch.cat((past_key, key), dim=-1)
            # (batch_size, num_head, sql_len+1, head_features)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))

        ''' 
        for first iteration: the query、key、value seq_len is seq_len 
        
        for second or others iteration: the query、key、value seq_len is 1
        # query: (batch_size, num_head, 1, head_features)
        # key: (batch_size, num_head, head_features, sql_len+1)
        # value: (batch_size, num_head, sql_len+1, head_features)
        # w: (batch_size, num_head, 1, sql_len+1)
        '''
        w = torch.matmul(query, key)
        w = w / (float(value.size(-1)) ** 0.5)

        # masked
        nd, ns = w.size(-2), w.size(-1)
        mask = self.masked[:, :, ns - nd: ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        # process attention mask
        if attention_mask is not None:
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # attention_output: (batch_size, num_head, sql_len, head_features)
        attention_output = torch.matmul(w, value)
        attention_output = self.merge_heads(attention_output)
        attention_output = self.linear(attention_output)
        attention_output = self.resid_dropout(attention_output)

        return attention_output, present


class MLP(nn.Module):
    def __init__(self, inner_dim, config):
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(config.n_embd, inner_dim)
        self.linear_2 = nn.Linear(inner_dim, config.n_embd)
        self.gelu = nn.GELU()

        nn.init.normal_(self.linear_1.weight, std=0.02)
        nn.init.normal_(self.linear_2.weight, std=0.02)

    def forward(self, inputs):
        outputs = self.gelu(self.linear_1(inputs))
        outputs = self.linear_2(outputs)
        return outputs


class Block(nn.Module):
    def __init__(self, n_ctx, config):
        super(Block, self).__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.layer_norm_1 = nn.LayerNorm(hidden_size, config.layer_norm_epsilon)
        self.attention = MaskedMultiHeadAttention(hidden_size, n_ctx, config)
        self.layer_norm_2 = nn.LayerNorm(hidden_size, config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)

    def forward(self, hidden_states, layer_past=None, attention_mask=None):
        # first: Masked Multi Head Self Attention
        hidden_states = self.layer_norm_1(hidden_states)
        attention_output, present = self.attention(hidden_states,
                                                   layer_past=layer_past,
                                                   attention_mask=attention_mask)
        hidden_states = hidden_states + attention_output

        # second: MLP layer
        hidden_states = self.layer_norm_2(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = hidden_states + mlp_out

        return hidden_states, present


class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        # initialize word embedding and position embedding
        self.word_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        init.normal_(self.word_embedding.weight, std=0.02)
        self.position_embedding = nn.Embedding(config.n_positions, config.n_embd)
        init.normal_(self.position_embedding.weight, std=0.01)
        self.drop_out = nn.Dropout(config.embd_pdrop)

        # transformer decoder block
        self.blocks = nn.ModuleList([Block(config.n_ctx, config) for _ in range(config.n_layer)])
        self.layer_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        else:
            raise ValueError("input_ids is None")

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.blocks)
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0

        # embedding layer
        word_embeds = self.word_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = word_embeds + position_embeds
        hidden_states = self.drop_out(hidden_states)

        # transformer block layer
        presents = []
        for i, (block, layer_past) in enumerate(zip(self.blocks, past_key_values)):
            hidden_states, present = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.layer_norm(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return {
            'hidden_states': hidden_states.view(*output_shape),
            'past_key_values': presents
        }


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__()
        self.gpt = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self,
                input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                lm_labels=None):
        hidden_states, presents = self.gpt(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values)

        lm_logits = self.lm_head(hidden_states)

        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, lm_logits, presents
        return {
            'logits': lm_logits,
            'past_key_values': presents
        }
