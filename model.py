SRC_EMB_SIZE = 64
ACTION_EMB_SIZE = 32
FIELD_EMB_SIZE = 8
LSTM_HIDDEN_DIM = 200
ATT_SIZE = 200
DROPOUT = 0.2


class TranxParser(nn.Module):
    def __init__(self, vocab, transition_system):
        super().__init__()
        self.vocab = vocab
        self.transition_system = transition_system
        # embeddings:
        # src text
        self.src_emb = nn.Embedding(len(vocab.source), SRC_EMB_SIZE, padding_idx=0)
        self.apply_const_and_reduce_emb = nn.Embedding(len(transition_system.grammar) + 1, ACTION_EMB_SIZE)
        self.primitives_emb = nn.Embedding(len(vocab.primitive), ACTION_EMB_SIZE)
        self.fields_emb = nn.Embedding(len(transition_system.grammar.fields), FIELD_EMB_SIZE)

        # encoder
        self.encoder = nn.LSTM(input_size=SRC_EMB_SIZE, 
                               hidden_size=LSTM_HIDDEN_DIM // 2, 
                               num_layers=1,
                               bidirectional=True,
                               dropout=0.2,
                               batch_first=True)

        # decoder
        self.decoder_input_dim = ACTION_EMB_SIZE + FIELD_EMB_SIZE + LSTM_HIDDEN_DIM + ATT_SIZE
        self.decoder = nn.LSTMCell(input_size=self.decoder_input_dim, hidden_size=LSTM_HIDDEN_DIM)
        
        self.lin_attn = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, bias=False)
        self.attention_vector_lin = nn.Linear(LSTM_HIDDEN_DIM * 2, ATT_SIZE, bias=False)

        # post decode
        self.ptr_net_lin = nn.Linear(LSTM_HIDDEN_DIM, ATT_SIZE, bias=False)
        self.applyconstrprob_lin = nn.Linear(ACTION_EMB_SIZE, ATT_SIZE, bias=False)
        self.query_vec_to_action_embed = nn.Linear(ATT_SIZE, SRC_EMB_SIZE, bias=False)
        self.gen_vs_copy_lin = nn.Linear(ATT_SIZE, 1)
        
        
    def encode(self, sents, sent_lens): # done
        # batch: 
        padded_sents = rnn_utils.pad_sequence(sents, batch_first=True)
#         print(padded_sents)
        embeddings = self.src_emb(padded_sents)
        print(embeddings.shape) # B x T x embdim
        inputs = rnn_utils.pack_padded_sequence(embeddings, sent_lens, batch_first=True)
        encodings, final_state = self.encoder(inputs)
        encodings, lens = rnn_utils.pad_packed_sequence(encodings, batch_first=True)
        print(encodings.shape) # B x T x hiddendim
        return encodings, final_state
    
    
    def get_prev_action_embs(self, batch, time_step, states_sequence):
        # TODO
        zeros_emb = torch.zeros(ACTION_EMB_SIZE)
        action_embs_prev = []
        parent_states = []
        for eid, example in enumerate(batch):
            # action t - 1
            if time_step < len(example.tgt_actions):
                parent_time_step = example.tgt_actions[time_step].parent_t
                prev_action = example.tgt_actions[time_step - 1].action
                if isinstance(prev_action, ApplyRuleAction):
                    action_emb = self.productions_emb.weight[self.grammar.prod2id[prev_action.production]]
                elif isinstance(prev_action, ReduceAction):
                    action_emb = self.productions_emb.weight[len(self.grammar)]
                else:
                    action_emb = self.primitives_emb.weight[self.vocab.primitive[prev_action.token]]
            else:
                action_emb = zeros_emb
                parent_time_step = 0
            action_embs_prev.append(action_emb)
            parent_states.append(states_sequence[parent_time_step][eid])
        return torch.stack(action_embs_prev), torch.stack(parent_states)
    
    
    def get_token_mask(self, sent_lens):
        # returns mask:  B x S, 1 where entries are to be masked, 0 for valid ones
        mask = torch.zeros((len(sent_lens), self.S))
        for ei in range(len(sent_lens)):
            mask[ei, sent_lens[ei]:] = 1
#         print(mask.byte())
        return mask.byte()


    def s_attention(self, encodings, encodings_attn, src_mask, h):
        # encdoings: B x S x hiddendim, h: B x hiddendim
        att_weight = torch.matmul(encodings_attn, h.unsqueeze(2)).squeeze(2) 
#         print(att_weight.shape) # B x S
        # src_mask has 1 where pad, fill -inf there
        att_weight.masked_fill_(src_mask, -float('inf'))
#             print(att_weight, src_mask)
        att_weight = F.softmax(att_weight, dim=1)
        batch_size, src_len = att_weight.shape
        ctx = torch.matmul(att_weight.view(batch_size, 1, src_len), encodings).squeeze(1)
#         print(ctx.shape) # B x hiddendim
        # s_att_prev is not previous in this time step, but the next step
        s_att = F.tanh(self.attention_vector_lin(torch.cat([ctx, h], 1)))
        return s_att
    
    def decode(self, batch, src_mask, encodings, final_state):
        # produce attention vectors
        batch_size = len(batch)
        h, c = final_state
#         print(h.shape)
        h, c = h.view(batch_size, -1), c.view(batch_size, -1)
#         print(h.shape) B x hiddendim
        
        inp = torch.zeros(batch_size, self.decoder_input_dim)
        states_sequence, s_att_all = [], []
        encodings_attn = self.lin_attn(encodings)
#         print(encodings_attn.shape) # same dim as encodings
        for t in range(self.T):
            if t > 0:
                # [act prev : ~s prev : pt]
                action_emb_prev, parent_states = self.get_prev_action_embs(batch, t, states_sequence)
                frontier = [self.grammar.field2id[e.tgt_actions[t].frontier_field] if t < len(e.tgt_actions) else 0 for
                            e in batch]
                nft = self.field_embed(torch.Tensor(frontier))
                inp = torch.cat([action_emb_prev, s_att_prev, nft, parent_states], dim=-1)
            
            h, c = self.decoder(inp, (h, c))
            states_sequence.append(h)

            # compute the combined attention 
            s_att_prev = self.s_attention(encodings, encodings_attn, src_mask, h)
#             print(s_att_prev.shape, "~s") # B x hiddendim
            s_att_all.append(s_att_prev)

        s_att_all = torch.stack(s_att_all, dim=0)
        return s_att_all  
        
    
    def forward(self, batch):
        # batch is list of examples
#         self.src_mask = self.get_token_mask(batch)
        self.T = max(len(e.tgt_actions) for e in batch)
        self.sents = [e.src_token_ids for e in batch]
        self.sent_lens = [len(s) for s in self.sents]
        self.S = max(self.sent_lens)
        print(self.T, self.S, self.sent_lens)
        sent_idxs = sorted(list(range(len(self.sents))), key=lambda i: -self.sent_lens[i])
        self.sents_sorted = [self.sents[i] for i in sent_idxs]
        self.sents_lens_sorted = [self.sent_lens[i] for i in sent_idxs]
        self.examples_sorted = [batch[i] for i in sent_idxs]
#         return sents_sorted
        print(self.sents_sorted, self.sents_lens_sorted)
        self.src_mask = self.get_token_mask(self.sents_lens_sorted)
        encodings, final_states = self.encode(self.sents_sorted, self.sents_lens_sorted)
        s_att_vecs = self.decode(self.examples_sorted, self.src_mask, encodings, final_states)
