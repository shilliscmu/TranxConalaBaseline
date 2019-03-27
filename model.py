import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.utils.rnn as rnn_utils

from action_info import ActionInfo
from hypothesis import Hypothesis
from transitions import ApplyRuleAction, ReduceAction, GenTokenAction

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
        self.grammar = self.transition_system.grammar
        # embeddings:
        # src text
        self.src_emb = nn.Embedding(len(vocab.source), SRC_EMB_SIZE, padding_idx=0)
        self.apply_const_and_reduce_emb = nn.Embedding(len(transition_system.grammar) + 1, ACTION_EMB_SIZE)
        # self.apply_const_and_reduce_emb = nn.Embedding(len(transition_system.grammar) + 1, ACTION_EMB_SIZE)
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

        #source_encodings_attention_linear_layer
        self.lin_attn = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, bias=False)
        self.attention_vector_lin = nn.Linear(LSTM_HIDDEN_DIM * 2, ATT_SIZE, bias=False)

        # post decode
        self.ptr_net_lin = nn.Linear(LSTM_HIDDEN_DIM, ATT_SIZE, bias=False)
        self.applyconstrprob_lin = nn.Linear(ACTION_EMB_SIZE, ATT_SIZE, bias=False)
        # self.attn_vec_to_action_emb = nn.Linear(ATT_SIZE, ACTION_EMB_SIZE, bias=False)
        self.attn_vec_to_action_emb = nn.Linear(ACTION_EMB_SIZE, ATT_SIZE, bias=False)
        self.gen_vs_copy_lin = nn.Linear(ATT_SIZE, 1)

    def parse(self, sentence, context=None, beam_size=5):
        primitive_vocab = self.vocab.primitive
        processed_sentence = self.process([sentence])
        source_encodings, (last_encoder_state, last_encoder_cell) = self.encode(processed_sentence, [len(sentence)])
        self.decoder_cell_initializer_linear_layer = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM)
        self.decoder_cell_initializer_linear_layer = self.decoder_cell_initializer_linear_layer.cuda()
        last_encoder_cell = last_encoder_cell.cuda()
        print("last encoder cell size: " + repr(last_encoder_cell.size()))
        print("last encoder cell state" + repr(last_encoder_state.size()))
        h_t1 = torch.tanh(self.decoder_cell_initializer_linear_layer(last_encoder_cell))

        hypothesis_scores = Variable(torch.cuda.FloatTensor([0.]), volatile=True)
        source_token_positions_by_token = OrderedDict()
        for token_position, token in enumerate(sentence):
            source_token_positions_by_token.setdefault(token, []).append(token_position)
        t = 0
        hypotheses = [Hypothesis()]
        hypotheses_states = [[]]
        finished_hypotheses = []

        while len(finished_hypotheses) < 15 and t < 100:
            num_of_hypotheses = len(hypotheses)
            expanded_source_encodings = source_encodings.expand(num_of_hypotheses, source_encodings.size(1),
                                                                source_encodings.size(2))
            expanded_source_encodings_attention_linear_layer = self.lin_attn.expand(
                num_of_hypotheses, self.lin_attn.size(1), self.lin_attn.size(2))
            if t == 0:
                x = Variable(torch.cuda.FloatTensor(1, 128).zero_(), volatile=True)
            else:
                actions = [h.actions[-1] for h in hypotheses]
                action_embeddings = []
                for action in actions:
                    if action:
                        action_embeddings.append(self.action_emb_from_action(action))
                    else:
                        action_embeddings.append(Variable(torch.cuda.FloatTensor(128).zero_()))
                action_embeddings = torch.stack(action_embeddings)
                encoder_inputs = [action_embeddings]
                encoder_inputs.append(att_t1)

                frontier_fields = [h.frontier_field.field for h in hypotheses]
                frontier_field_embeddings = self.fields_emb(
                    Variable(torch.cuda.FloatTensor([self.grammar.field_to_id[f] for f in frontier_fields])))
                encoder_inputs.append(frontier_field_embeddings)

                parent_created_times = [h.frontier_node.created_time for h in hypotheses]
                parent_states = torch.stack(
                    [hypotheses_states[h_id][parent_created_time][0]] for h_id, parent_created_time in
                    enumerate(parent_created_times))
                parent_cells = torch.stack(
                    [hypotheses_states[h_id][parent_created_time][1] for h_id, parent_created_time in
                     enumerate(parent_created_times)])
                encoder_inputs.append(parent_states)

                x = torch.cat(encoder_inputs, dim=-1)
            (h_t, cell), attention = self.step(x, h_t1, expanded_source_encodings, expanded_source_encodings_attention_linear_layer)
            log_p_of_each_apply_rule_action = F.log_softmax(self.production_prediction(attention), dim=-1)
            p_of_generating_each_primitive_in_vocab = F.softmax(self.primitive_prediction(attention), dim=-1)
            p_of_copying_from_source_sentence = self.ptr_net_lin(source_encodings, None, attention.unsqueeze(0).squeeze(0))
            p_of_making_primitive_prediction = F.softmax(self.gen_vs_copy_lin(attention), dim=-1)
            p_of_each_primitive = p_of_making_primitive_prediction[:, 0].unsqueeze(1) * p_of_generating_each_primitive_in_vocab

            hypothesis_ids_for_which_we_gentoken = []
            hypothesis_unknowns_resulting_from_gentoken = []
            hypothesis_ids_for_which_we_applyrule = []
            hypothesis_production_ids_resulting_from_applyrule_actions = []
            hypothesis_scores_resulting_from_applyrule_actions = []

            for hypothesis_id, hypothesis in enumerate(hypotheses):
                action_types = self.transition_sys.get_valid_continuation_types(hypothesis)
                for action_type in action_types:
                    if action_type == ApplyRuleAction:
                        productions = self.transition_sys.get_valid_continuating_productions(hypothesis)
                        for production in productions:
                            production_id = self.grammar.production_to_id[production]
                            hypothesis_production_ids_resulting_from_applyrule_actions.append(production_id)
                            production_score = log_p_of_each_apply_rule_action[hypothesis_id, production_id].data[0]
                            new_hypothesis_score = hypothesis.score + production_score
                            hypothesis_scores_resulting_from_applyrule_actions.append(new_hypothesis_score)
                            hypothesis_ids_for_which_we_applyrule.append(hypothesis_id)
                    elif action_type == ReduceAction:
                        reduce_score = log_p_of_each_apply_rule_action[hypothesis_id, len(self.grammar)].data[0]
                        new_hypothesis_score = hypothesis.score + reduce_score
                        hypothesis_scores_resulting_from_applyrule_actions.append(new_hypothesis_score)
                        hypothesis_production_ids_resulting_from_applyrule_actions.append(len(self.grammar))
                        hypothesis_ids_for_which_we_applyrule.append(hypothesis_id)
                    else:
                        hypothesis_ids_for_which_we_gentoken.append(hypothesis_id)
                        hypothesis_copy_probabilities_by_token = dict()
                        copied_unks_info = []
                        for token, token_positions in source_token_positions_by_token.items():
                            total_copy_prob = torch.gather(p_of_copying_from_source_sentence[hypothesis_id], 0, Variable(torch.cuda.LongTensor(token_positions))).sum()
                            p_of_making_copy = p_of_making_primitive_prediction[hypothesis_id, 1] * total_copy_prob
                            if token in primitive_vocab:
                                token_id = primitive_vocab[token]
                                p_of_each_primitive[hypothesis_id, token_id] = p_of_each_primitive[hypothesis_id, token_id] + p_of_making_copy
                                hypothesis_copy_probabilities_by_token[token] = (token_positions, p_of_making_copy.data[0])
                            else:
                                copied_unks_info.append({'token': token, 'token_positions': token_positions, 'copy_prob': p_of_making_copy.data[0]})
                        if len(copied_unks_info) > 0:
                            copied_unk = np.array([unk['copy_prob'] for unk in copied_unks_info]).argmax()
                            copied_token = copied_unks_info[copied_unk]['token']
                            p_of_each_primitive[hypothesis_id, primitive_vocab.unk_id] = copied_unks_info[copied_unk]['copy_prob']
                            hypothesis_unknowns_resulting_from_gentoken.append(copied_token)
                            hypothesis_copy_probabilities_by_token[copied_token] = (copied_unks_info[copied_unk]['token_positions'], copied_unks_info[copied_unk]['copy_prob'])

            new_hypothesis_scores = None
            if hypothesis_scores_resulting_from_applyrule_actions:
                new_hypothesis_scores = Variable(torch.cuda.FloatTensor(hypothesis_scores_resulting_from_applyrule_actions))
            if hypothesis_ids_for_which_we_gentoken:
                log_p_of_each_primitive = torch.log(p_of_each_primitive)
                gen_token_new_hypothesis_scores = (hypothesis_scores[hypothesis_ids_for_which_we_gentoken].unsqueeze(1) + log_p_of_each_primitive[hypothesis_ids_for_which_we_gentoken, :]).view(-1)

                if new_hypothesis_scores is None:
                    new_hypothesis_scores = gen_token_new_hypothesis_scores
                else:
                    new_hypothesis_scores = torch.cat([new_hypothesis_scores, gen_token_new_hypothesis_scores])
            top_new_hypothesis_scores, top_new_hypothesis_positions = torch.topk(new_hypothesis_scores, k=min(new_hypothesis_scores.size(0), 15 - len(finished_hypotheses)))

            working_hypothesis_ids = []
            new_hypotheses = []
            for new_hypothesis_score, new_hypothesis_position in zip(top_new_hypothesis_scores.data.cpu(), top_new_hypothesis_positions.data.cpu()):
                action_info = ActionInfo()
                if new_hypothesis_position < len(hypothesis_scores_resulting_from_applyrule_actions):
                    previous_hypothesis_id = hypothesis_ids_for_which_we_applyrule[new_hypothesis_position]
                    previous_hypothesis = hypotheses[previous_hypothesis_id]
                    production_id = hypothesis_scores_resulting_from_applyrule_actions[new_hypothesis_position]
                    if production_id < len(self.grammar):
                        apply_production = self.grammar.id_to_production[production_id]
                        action = ApplyRuleAction(apply_production)
                    else:
                        action = ReduceAction()
                else:
                    token_id = (new_hypothesis_position - len(hypothesis_scores_resulting_from_applyrule_actions)) % p_of_each_primitive.size(1)
                    previous_hypothesis_id = hypothesis_ids_for_which_we_gentoken[(new_hypothesis_position - len(hypothesis_scores_resulting_from_applyrule_actions)) // p_of_each_primitive.size(1)]
                    previous_hypothesis = hypotheses[previous_hypothesis_id]
                    if token_id == primitive_vocab.unk_id:
                        if hypothesis_unknowns_resulting_from_gentoken:
                            token = hypothesis_unknowns_resulting_from_gentoken[(new_hypothesis_position - len(hypothesis_scores_resulting_from_applyrule_actions)) // p_of_each_primitive.size(1)]
                        else:
                            token = primitive_vocab.id_to_word[primitive_vocab.unk_id]
                    else:
                        token = primitive_vocab.id_2_word[token_id]
                    action = GenTokenAction(token)

                    if token in source_token_positions_by_token:
                        action_info.copy_from_src = True
                        action_info.src_token_position = source_token_positions_by_token[token]

                action_info.action = action
                action_info.t = t
                if t > 0:
                    action_info.parent_t = previous_hypothesis.frontier_node.created_time
                    action_info.frontier_prod = previous_hypothesis.frontier_node.production
                    action_info.frontier_field = previous_hypothesis.frontier_field.field
                new_hypothesis = previous_hypothesis.clone_and_apply_action_info(action_info)
                new_hypothesis.score = new_hypothesis_score

                if new_hypothesis.completed:
                    finished_hypotheses.append(new_hypothesis)
                else:
                    new_hypotheses.append(new_hypothesis)
                    working_hypothesis_ids.append(previous_hypothesis_id)
            if working_hypothesis_ids:
                hypothesis_states = [hypothesis_states[i] + [(h_t[i], cell[i])] for i in working_hypothesis_ids]
                h_t1 = (h_t[working_hypothesis_ids], cell[working_hypothesis_ids])
                att_t1 = attention[working_hypothesis_ids]
                hypotheses = new_hypotheses
                hypothesis_scores = Variable(torch.cuda.FloatTensor([hyp.score for hyp in hypotheses]))
                t += 1
            else:
                break

        finished_hypotheses.sort(key=lambda hyp: -hyp.score)
        return finished_hypotheses

    def step(self, x, h_t1, expanded_source_encodings, expanded_source_encodings_attention_linear_layer):
        h_t, cell_t = self.decoder(x, h_t1)
        context_t = self.s_attention(expanded_source_encodings, expanded_source_encodings_attention_linear_layer, )
        attention = torch.tanh(self.attention_vector_lin(torch.cat([h_t, context_t], 1)))
        attention = self.dropout(attention)
        return (h_t, cell_t), attention

    def encode(self, sents, sent_lens):  # done
        # batch:

        padded_sents = rnn_utils.pad_sequence(sents, batch_first=True)
        #         print(padded_sents)
        print("About to embed source sentences.")
        embeddings = self.src_emb(padded_sents)
        print(embeddings.shape)  # B x T x embdim
        print("About to pad embedded sentences.")
        inputs = rnn_utils.pack_padded_sequence(embeddings, sent_lens, batch_first=True)
        print("About to encode embedded sentences.")
        inputs = inputs.cuda()
        encodings, final_state = self.encoder(inputs)
        encodings, lens = rnn_utils.pad_packed_sequence(encodings, batch_first=True)
        print(encodings.shape)  # B x T x hiddendim
        return encodings, final_state

    def action_emb_from_action(self, action):
        if isinstance(action, ApplyRuleAction):
            action_emb = self.productions_emb.weight[self.grammar.production_to_id[action.production]]
        elif isinstance(action, ReduceAction):
            action_emb = self.productions_emb.weight[len(self.grammar)]
        else:
            action_emb = self.primitives_emb.weight[self.vocab.primitive[action.token]]
        return action_emb

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
                action_emb = self.action_emb_from_action(self, prev_action)
            else:
                action_emb = zeros_emb
                parent_time_step = 0
            action_embs_prev.append(action_emb)
            # TODO: encoder_inputs.append(att_t1)
            # TODO: frontier_fields = [h.frontier_field.field for h in hypotheses]
            #                 frontier_field_embeddings = self.fields_emb(Variable(torch.cuda.FloatTensor([self.grammar.field_to_id[f] for f in frontier_fields])))
            #                 encoder_inputs.append(frontier_field_embeddings)
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
        if src_mask is not None:
            att_weight.masked_fill_(src_mask, -float('inf'))
        #             print(att_weight, src_mask)
        att_weight = F.softmax(att_weight, dim=1)
        batch_size, src_len = att_weight.shape
        ctx = torch.matmul(att_weight.view(batch_size, 1, src_len), encodings).squeeze(1)
        #         print(ctx.shape) # B x hiddendim
        # s_att_prev is not previous in this time step, but the next step
        s_att = torch.tanh(self.attention_vector_lin(torch.cat([ctx, h], 1)))
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
                frontier = [self.grammar.field_to_id[e.tgt_actions[t].frontier_field] if t < len(e.tgt_actions) else 0 for
                            e in batch]
                nft = self.fields_emb(torch.Tensor(frontier))
                inp = torch.cat([action_emb_prev, s_att_prev, nft, parent_states], dim=-1)

            print("cat'd previous action embeddings.")
            # print("input is cuda?" + repr(inp.is_cuda()))
            # print("h is cuda?" + repr(h.is_cuda()))
            # print("c is cuda?" + repr(c.is_cuda()))
            inp = inp.cuda()
            h, c = self.decoder(inp, (h, c))
            states_sequence.append(h)

            print("computed combined attention.")
            # compute the combined attention
            src_mask = src_mask.cuda()
            s_att_prev = self.s_attention(encodings, encodings_attn, src_mask, h)
            #             print(s_att_prev.shape, "~s") # B x hiddendim
            s_att_all.append(s_att_prev)

        s_att_all = torch.stack(s_att_all, dim=0)
        return s_att_all

    def pointer_weights(self, encodings, src_mask, s_att_vecs):
        # to compute hWs. encodings: B x hiddendim, s_att_vecs: T x B x  attsize, ptr lin layer dimx -> attsize
        hW = self.ptr_net_lin(encodings)  # B x S x attsize
        if len(s_att_vecs.shape) == 2:
            s_att_vecs.unsqueeze(0)  # T = 1
        scores = torch.matmul(s_att_vecs, hW.permute(0, 2,
                                                     1))  # hW is (B x S x attsize), s_att_vecs is (B x T x attsize|) or H x 1 x attsize
        scores = scores.permute(1, 0, 2)  # T x B x S
        # src_mask is B x S
        if src_mask is not None:
            src_token_mask = src_mask.unsqueeze(0).expand_as(scores)
            scores.masked_fill_(src_token_mask, -float('inf'))
        scores = scores.permute(1, 0, 2)
        if len(s_att_vecs.shape) == 2:
            scores = scores.squeeze(1)
        return F.softmax(scores, dim=-1)  # B x T x S or H x S

    # production_readout
    def get_action_prob(self, s_att_vecs, lin_layer, weight):
        # to compute aWs. s_att_vecs: T x B x  attsize, weight: |a| x embdim, Lin layer dimx -> attsize 
        print("getting action prob.")
        #weight.weight is 97x128
        aW = lin_layer(weight.weight)  # |a| x  attsize
        # aW is (|a| x  attsize), s_att_vecs is (T x B x  attsize)
        #aW is self.production_embed.weight
        scores = torch.matmul(s_att_vecs, aW.t())
        if len(scores.shape) == 3:
            scores = scores.permute(1, 0, 2)  # B x T x |a|
        return F.softmax(scores, dim=-1)  # B x T x |a| 

    def get_rule_masks(self, batch):
        is_applyconstr = torch.zeros((len(batch), self.T))
        is_gentoken = torch.zeros((len(batch), self.T))
        is_copy = torch.zeros((len(batch), self.T))
        applyconstr_ids = torch.zeros((len(batch), self.T), dtype=torch.int64)
        gentok_ids = torch.zeros((len(batch), self.T), dtype=torch.int64)
        is_copy_tok = torch.zeros((len(batch), self.T, self.S))
        for ei, example in enumerate(self.examples_sorted):
            for t in range(self.T):
                if t < len(example.tgt_actions):
                    action = example.tgt_actions[t].action

                    if isinstance(action, ApplyRuleAction):
                        is_applyconstr[ei, t] = 1
                        applyconstr_ids[ei, t] = self.grammar.production_to_id[action.production]

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
        is_applyconstr, is_gentoken, is_copy, applyconstr_ids, gentok_ids, is_copy_tok = [e.cuda() for e in self.get_rule_masks(batch)]

        p_a_apply = self.get_action_prob(s_att_vecs, self.attn_vec_to_action_emb,
                                         self.apply_const_and_reduce_emb)  # B x T x |a|
        p_gen = self.gen_vs_copy_lin(s_att_vecs)  # T x B x 1
        p_copy = 1 - p_gen  # T x B x 1
        src_mask = src_mask.cuda()
        p_v_copy = self.pointer_weights(encodings, src_mask, s_att_vecs)  # B x T x S
        p_tok_gen = self.get_action_prob(s_att_vecs, self.attn_vec_to_action_emb, self.primitives_emb)  # B x T x |t|

        print("p_v_copy type: " + p_v_copy.type())
        print("is_copy_tok type: " + is_copy_tok.type())
        target_p_copy = torch.sum(p_v_copy * is_copy_tok, dim=2).t()  # T x B

        # target lookup
        # T x B (*ids is BxT)
        print("index size: " + repr(applyconstr_ids.t().unsqueeze(2).permute(1,0,2).size()))
        print("p_a_apply size: " + repr(p_a_apply.size()))
        print("applyconstr_ids size: " + repr(applyconstr_ids.size()))
        target_p_a_apply = torch.gather(p_a_apply, dim=2, index=applyconstr_ids.t().unsqueeze(2).permute(1,0,2))
        print("index size: " + repr(gentok_ids.t().unsqueeze(2).permute(1,0,2).size()))
        print("p_tok_gen size: " + repr(p_tok_gen.size()))
        print("gentok_ids size: " + repr(gentok_ids.size()))
        target_p_a_gen = torch.gather(p_tok_gen, dim=2, index=gentok_ids.t().unsqueeze(2).permute(1,0,2))

        # T x B
        print("target_p_a_apply size: " + repr(target_p_a_apply.squeeze(1).size()))
        print("is_applyconstr.t() size: " + repr(is_applyconstr.t().size()))
        p_a_apply_target = target_p_a_apply.squeeze(1) * is_applyconstr.t()
        print("\np_gen[:,:,0] size: " + repr(p_gen[:, :, 0].size()))
        print("target_p_a_gen size: " + repr(target_p_a_gen.squeeze(1).size()))
        print("is_gentoken.t() size: " + repr(is_gentoken.t().size()))
        p_a_gen_target = p_gen[:, :, 0] * target_p_a_gen.squeeze(1) * is_gentoken.t()  # is_... is B x T
        p_a_gen_target += p_copy[:, :, 0] * target_p_copy * is_copy.t()  # T x B
        action_prob_target = p_a_apply_target + p_a_gen_target

        action_mask_sum = is_applyconstr + is_gentoken + is_copy
        action_mask_sum = action_mask_sum.t()  # T x B

        # 1s where we pad, these we can ignore
        action_mask_pad = action_mask_sum == 0

        action_prob_target.masked_fill_(action_mask_pad, 1e-7)
        # make the not so useful stuff 0
        print("action_prob_target.log type: " + action_prob_target.log().type())
        # print("action_mask_pad type: " + action_mask_pad.type('torch.FloatTensor')).type()
        action_prob_target = action_prob_target.log() * (1 - action_mask_pad.type('torch.cuda.FloatTensor'))

        return torch.sum(action_prob_target, dim=0)  # B

    def forward(self, batch):
        # batch is list of examples
        #         self.src_mask = self.get_token_mask(batch)
        self.T = max(len(e.tgt_actions) for e in batch)
        self.sents = [self.process(e.sentence) for e in batch]
        # print("sents len: " + repr(len(self.sents)))
        # self.sents = torch.cuda.LongTensor(self.sents)
        self.sent_lens = [len(s) for s in self.sents]
        self.S = max(self.sent_lens)
        print(self.T, self.S, self.sent_lens)
        sent_idxs = sorted(list(range(len(self.sents))), key=lambda i: -self.sent_lens[i])
        self.sents_sorted = [self.sents[i] for i in sent_idxs]
        self.sents_lens_sorted = [self.sent_lens[i] for i in sent_idxs]
        self.examples_sorted = [batch[i] for i in sent_idxs]
        #         return sents_sorted
        print(self.sents_sorted, self.sents_lens_sorted)
        # list_of_tensor_sents = [torch.cuda.LongTensor(sent) for sent in self.sents_sorted]
        # print("tensor of tensor sents type: " + torch.stack(list_of_tensor_sents).type())
        self.src_mask = self.get_token_mask(self.sents_lens_sorted)
        # self.sents_sorted = torch.cuda.LongTensor(self.sents_sorted)
        # self.sents_lens_sorted = torch.cuda.LongTensor(self.sents_lens_sorted)

        encodings, final_states = self.encode(self.sents_sorted, self.sents_lens_sorted)
        self.src_mask = self.src_mask.cuda()
        s_att_vecs = self.decode(self.examples_sorted, self.src_mask, encodings, final_states)
        print("Finshed decode.")
        scores = self.compute_target_probabilities(encodings, s_att_vecs, self.src_mask, self.examples_sorted)
        print("Finshed scoring.")
        return scores, final_states[0]

    def process(self, sentence):
        source = self.vocab.source
        if isinstance(sentence[0], (list,)):
            processed_sent = []
            for sent in sentence:
                processed_sent.append(torch.cuda.LongTensor([source.__getitem__(word) for word in sent]))
            return processed_sent
        else:
            word_ids = [source.__getitem__(word) for word in sentence]
            return torch.cuda.LongTensor(word_ids)

    def save(self, path, saveGrammar):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if saveGrammar:
            params = {
                'transition_system': self.transition_system,
                'vocab': self.vocab,
                'state_dict': self.state_dict()
            }
        else:
            params = {
                'state_dict': self.state_dict()
            }
        torch.save(params, path)