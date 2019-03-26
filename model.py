import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from transitions import ApplyRuleAction, ReduceAction


SRC_EMB_SIZE = 128
ACTION_EMB_SIZE = SRC_EMB_SIZE
FIELD_EMB_SIZE = 64
LSTM_HIDDEN_DIM = 256
ATT_SIZE = 256
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
        self.attn_vec_to_action_emb = nn.Linear(ATT_SIZE, ACTION_EMB_SIZE, bias=False)
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
        print(src_mask.shape, 'mask')
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
    
    
    def pointer_weights(self, encodings, src_mask, s_att_vecs):
        # to compute hWs. encodings: B x hiddendim, s_att_vecs: T x B x  attsize, ptr lin layer dimx -> attsize
        hW = self.ptr_net_lin(encodings) # B x S x attsize
        scores = torch.matmul(s_att_vecs, hW.permute(0, 2, 1))  # hW is (B x S x attsize), att_vecs is (B x T x attsize|)
        scores = scores.permute(1, 0, 2) # T x B x S
        # src_mask is B x S
        src_token_mask = self.src_mask.unsqueeze(0).expand_as(scores)
        scores.masked_fill_(src_token_mask, -float('inf'))
        return F.softmax(scores.permute(1, 0, 2), dim=2) # B x T x S 

    def get_action_prob(self, s_att_vecs, lin_layer, weight):
        # to compute aWs. s_att_vecs: T x B x  attsize, weight: |a| x embdim, Lin layer dimx -> attsize 
        aW = lin_layer(weight.weight) # |a| x  attsize
        # aW is (|a| x  attsize), s_att_vecs is (T x B x  attsize)
        scores = torch.matmul(s_att_vecs, aW.t())  
        scores = scores.permute(1, 0, 2) # B x T x |a| 
        return F.softmax(scores, dim=2)  # B x T x |a| 
        
    def get_rule_masks(self, batch):
        is_applyconstr = torch.zeros((len(batch), self.T))
        is_gentoken = torch.zeros((len(batch), self.T))
        is_copy = torch.zeros((len(batch), self.T))
        applyconstr_ids = torch.zeros((len(batch), self.T))
        gentok_ids = torch.zeros((len(batch), self.T))
        is_copy_tok = torch.zeros((len(batch), self.T, self.S))
        for ei, example in enumerate(self.examples):
            for t in range(self.T):
                if t < len(example.tgt_actions):
                    action = example.tgt_actions[t].action

                    if isinstance(action, ApplyRuleAction):
                        is_applyconstr[ei, t] = 1
                        applyconstr_ids[ei, t] = self.grammar.prod2id[action.production]

                    elif isinstance(action, ReduceAction):
                        is_applyconstr[ei, t] = 1
                        applyconstr_ids[ei, t] = len(self.grammar)

                    else:  # not apply constr
                        src_sent = self.src_sents[ei]
                        action_token = str(action.token)
                        token_idx = self.vocab.primitive[action.token]
                        gentok_ids[ei, t] = token_idx
                        no_copy = True
                        for idx, src_tok in enumerate(src_sent):
                            if src_tok == action_token:
                                is_copy_tok[ei, t, idx] = 1
                                no_copy = False
                                is_copy[ei, t] = 1
                        if no_copy or token_idx != self.vocab.primitive.unk_id:
                            is_gentoken[ei, t] = 1

        return is_applyconstr, is_gentoken, is_copy, applyconstr_ids, gentok_ids, is_copy_tok
    
      
    def compute_target_probabilities(self, encodings, s_att_vecs, src_mask, batch):
        # s_att_vecs is T x B x attsize
        is_applyconstr, is_gentoken, is_copy, applyconstr_ids, gentok_ids, is_copy_tok = self.get_rule_masks(batch)

        p_a_apply = self.get_action_prob(s_att_vecs, self.attn_vec_to_action_emb,
                                         self.apply_const_and_reduce_emb)  # B x T x |a|
        p_gen = self.gen_vs_copy_lin(s_att_vecs)  # T x B x 1
        p_copy = 1 - p_gen  # T x B x 1
        p_v_copy = self.pointer_weights(encodings, src_mask, s_att_vecs)  # B x T x S
        p_tok_gen = self.get_action_prob(s_att_vecs, self.attn_vec_to_action_emb, self.primitives_emb)  # B x T x |t|

        target_p_copy = torch.sum(p_v_copy * is_copy_tok, dim=2).t()  # T x B
        # target lookup
        # T x B (*ids is BxT)
        target_p_a_apply = torch.gather(p_a_apply, dim=2, index=applyconstr_ids.t().unsqueeze(2))
        target_p_a_gen = torch.gather(p_tok_gen, dim=2, index=gentok_ids.t().unsqueeze(2))

        # T x B
        p_a_apply_target = target_p_a_apply * is_applyconstr.t()
        p_a_gen_target = p_gen[:, :, 0] * target_p_a_gen * is_gentoken.t()  # is_... is B x T
        p_a_gen_target += p_copy[:, :, 0] * target_p_copy * is_copy.t()  # T x B
        action_prob_target = p_a_apply_target + p_a_gen_target

        action_mask_sum = is_applyconstr + is_gentoken + is_copy
        action_mask_sum = action_mask_sum.t()  # T x B

        # 1s where we pad, these we can ignore
        action_mask_pad = action_mask_sum == 0

        action_prob_target.masked_fill_(action_mask_pad, 1e-7)
        # make the not so useful stuff 0
        action_prob_target = action_prob_target.log() * (1 - action_mask_pad)

        return torch.sum(action_prob_target, dim=0)  # B

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
        scores = self.compute_target_probabilities(encodings, s_att_vecs, self.src_mask, self.examples_sorted)
        return scores, final_states[0]

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'transition_system': self.transition_system,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

    @classmethod
    def load(cls, model_path):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        transition_system = params['transition_system']
        # update saved args
        saved_state = params['state_dict']
        saved_args.cuda = cuda

        parser = cls(vocab, transition_system)

        parser.load_state_dict(saved_state)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        parser.to(device)
        parser.eval()

        return parser
