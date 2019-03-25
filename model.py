import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from asdl.transition_system import ApplyRuleAction, ReduceAction

SRC_EMB_SIZE = 64
ACTION_EMB_SIZE = 32
PROD_EMB_SIZE = 16
# TYPE_EMB_SIZE = 16
FIELD_EMB_SIZE = 8
LSTM_HIDDEN_DIM = 200
ATT_SIZE = 200
DROPOUT = 0.2


class TranxParser(nn.Module):
    def __init__(self, vocab, transition_system):
        super().__init__()
        # embeddings for src vocab, primitive tokens, asdl types, fields
        self.src_emb = nn.Embedding(len(vocab.source), SRC_EMB_SIZE)
        self.productions_emb = nn.Embedding(len(transition_system.grammar) + 1, PROD_EMB_SIZE)
        self.primitives_emb = nn.Embedding(len(vocab.primitive), ACTION_EMB_SIZE)
        self.fields_emb = nn.Embedding(len(transition_system.grammar.fields), FIELD_EMB_SIZE)

        self.dropout = nn.Dropout(DROPOUT)

        # encoder
        self.encoder = nn.LSTM(input_size=SRC_EMB_SIZE, hidden_size=LSTM_HIDDEN_DIM // 2, num_layers=1,
                               bidirectional=True,
                               dropout=0.2)

        # decoder
        self.decoder_input_dim = ACTION_EMB_SIZE + FIELD_EMB_SIZE + LSTM_HIDDEN_DIM + ATT_SIZE
        self.decoder = nn.LSTMCell(input_size=self.decoder_input_dim, hidden_size=LSTM_HIDDEN_DIM)
        self.lin_attn = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, bias=False)
        self.attention_vector_lin = nn.Linear(LSTM_HIDDEN_DIM * 2, ATT_SIZE, bias=False)
        self.ptr_net_lin = nn.Linear(LSTM_HIDDEN_DIM, ATT_SIZE, bias=False)
        self.applyconstrprob_lin = nn.Linear(PROD_EMB_SIZE, ATT_SIZE, bias=False)

        self.query_vec_to_action_embed = nn.Linear(ATT_SIZE, SRC_EMB_SIZE, bias=False)
        self.gen_vs_copy_lin = nn.Linear(ATT_SIZE, 1)

    def forward(self, batch, sentences, sentence_lens):
        encodings, final_states, encoding_atts = self.encode(sentences, sentence_lens)
        attn_vectors = self.decode(batch, encodings, encoding_atts, final_states)

        # TODO 

        return scores, final_states[0]

    def encode(self, sentences, sentence_lens):
        sentences_embs = self.src_emb(sentences)
        packed_embs = rnn_utils.pack_padded_sequence(sentences_embs, sentence_lens)
        encodings, final_states = self.encoder(packed_embs)
        encodings, _ = rnn_utils.pad_packed_sequence(encodings)
        encodings = encodings.permute(1, 0, 2)  # make batch first
        return encodings, final_states, self.lin_attn(encodings)

    def decode(self, batch, encodings, encodings_attn, final_states):
        # st = fLSTM([act prev : ~s prev : pt],s prev), ~s = tanh(Wc[c : s]). c is context vec
        h0, c0 = final_states
        h = h0.view(-1, LSTM_HIDDEN_DIM)
        c = c0.view(-1, LSTM_HIDDEN_DIM)

        batch_size = len(batch)

        att_vecs = []
        states_sequence = []

        input = torch.zeros((batch_size, self.decoder_input_dim))

        for t in range(batch.max_action_num):
            # LSTM cell input: [act prev : ~s prev : pt]
            if t > 0:
                action_emb_prev, parent_states = self.get_prev_action_embs(batch, t, states_sequence)
                nft = self.field_embed(batch.get_frontier_field_idx(t))
                input = torch.cat([action_emb_prev, s_att_prev, nft, parent_states], dim=-1)

            h, c = self.decoder(input, (h, c))

            # TODO
            att_weight = torch.matmul(encodings_attn, h.unsqueeze(2)).squeeze(2)
            if batch.src_token_mask is not None:
                att_weight.data.masked_fill_(batch.src_token_mask, -float('inf'))
            att_weight = F.softmax(att_weight, dim=-1)
            batch_size, src_len = att_weight.shape
            ctx = torch.matmul(att_weight.view(batch_size, 1, src_len), encodings).squeeze(
                1)  # encoding is (bs x src_len x hidden_dim)

            # s_att_prev is not previous in this time step, but the next step
            s_att_prev = F.tanh(self.attention_vector_lin(torch.cat([ctx, h], 1)))
            att_vecs.append(s_att_prev)
            states_sequence.append(h)

        att_vecs = torch.stack(att_vecs, dim=0)
        return att_vecs

    def get_prev_action_embs(self, batch, time_step, states_sequence):
        # TODO
        zeros_emb = torch.zeros(ACTION_EMB_SIZE)
        a_tm1_embeds = []
        parent_states = []
        for eid, example in enumerate(batch.examples):
            # action t - 1
            if time_step < len(example.tgt_actions):
                parent_time_step = example.tgt_actions[time_step].parent_t
                prev_action = example.tgt_actions[time_step - 1]
                if isinstance(prev_action.action, ApplyRuleAction):
                    a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[prev_action.action.production]]
                elif isinstance(prev_action.action, ReduceAction):
                    a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                else:
                    a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[prev_action.action.token]]
            else:
                a_tm1_embed = zeros_emb
                parent_time_step = 0
            a_tm1_embeds.append(a_tm1_embed)
        parent_states.append(states_sequence[parent_time_step][eid])
        return torch.stack(a_tm1_embeds), torch.stack(parent_states)

    def pointer_weights(self, encodings, src_token_mask, att_vecs):
        # (batch_size, 1, src_sent_len, query_vec_size)
        # hi W
        hW = self.ptr_net_lin(encodings).permute(0, 2, 1)  # .unsqueeze(1)
        att_vecs = att_vecs.permute(1, 0, 2)
        scores = torch.matmul(att_vecs, hW)  # hW is (b x |qv| x S), att_vecs is (b x T x |qv|)
        scores = scores.permute(1, 0, 2)

        # TODO mask

        return F.softmax(scores, dim=-1)


    def applyconstr_prob(self, src_token_mask, att_vecs):
        # ac W
        aW = self.applyconstrprob_lin(self.productions_emb.weight).permute(0, 2, 1)  # .unsqueeze(1)
        att_vecs = att_vecs.permute(1, 0, 2)
        scores = torch.matmul(att_vecs, aW)  # hW is (b x |qv| x S), att_vecs is (b x T x |qv|)
        scores = scores.permute(1, 0, 2)
        
        # TODO mask 
        return F.softmax(scores, dim=-1)
#
# model = TranxParser()
# print(model)
# print("number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
