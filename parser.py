from action_info import ActionInfo
from hypothesis import Hypothesis
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import TranxParser
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np

from transitions import ApplyRuleAction, ReduceAction, GenTokenAction

"""
Class for parsing utterance to AST 
"""

class Parser(nn.Module):
    def __init__(self, vocab, transition_sys):
        super(Parser, self).__init__()
        self.vocab = vocab
        self.transition_sys = transition_sys
        self.grammar = self.transition_sys.grammar

        self.fields_emb = nn.Embedding(len(transition_sys.grammar.fields), 64)
        nn.init.xavier_normal(self.fields_emb.weight.data)
        self.apply_const_and_reduce_emb = nn.Embedding(len(transition_sys.grammar) + 1, 128)
        nn.init.xavier_normal(self.apply_const_and_reduce_emb.weight.data)
        self.primitives_emb = nn.Embedding(len(vocab.primitive), 128)
        nn.init.xavier_normal(self.primitives_emb.weight.data)

        # action embeddings, field embeddings, hidden_state, attention
        input_dimension = 128 + 64 + 256 + 256
        self.decoder = nn.LSTMCell(input_dimension, 256)
        self.decoder_cell_initializer_linear_layer = nn.Linear(256, 256)
        self.lin_attn = nn.Linear(256, 256, bias=False)
        self.attention_vector_lin = nn.Linear(256 + 256, 256, bias=False)
        self.constructor_or_reduce_prediction_bias = nn.Parameter(torch.FloatTensor(len(transition_sys.grammar) + 1).zero_())
        self.token_generation_prediction_bias = nn.Parameter(torch.FloatTensor(self.vocab.primitive).zero_())

        # copy mechanism
        self.ptr_net_lin = PointerNet(256, 256)
        self.gen_vs_copy_lin = nn.Linear(256, 2)

        # use this to get action probs by dot-prod linearized attention vec and action embeds
        self.attn_vec_to_action_emb = nn.Linear(256, 128, bias=False)

        self.production_prediction = lambda att: F.linear(self.attn_vec_to_action_emb(att),
                                                          self.apply_const_and_reduce_emb.weight, self.constructor_or_reduce_prediction_bias)
        self.primitive_prediction = lambda att: F.linear(self.attn_vec_to_action_emb(att),
                                                         self.primitives_emb.weight, self.token_generation_prediction_bias)
        self.dropout = nn.Dropout(0.3)

    def process(self, sentences, vocab):
        if type(sentences[0]) == list:
            word_ids = [[vocab[word] for word in sentence] for sentence in sentences]
        else:
            word_ids = [vocab[word] for word in sentences]
        max_len = max(len(sentence) for sentence in word_ids)
        batch_size = len(word_ids)

        transposed_sentences = []
        for i in range(max_len):
            transposed_sentences.append(
                [word_ids[j][i] if len(word_ids[j]) > i else '<pad>' for j in range(batch_size)])

        processed_sentence = Variable(torch.cuda.LongTensor(transposed_sentences), volatile=False, requires_grad=False)
        return processed_sentence

    def parse(self, sentence):
        primitive_vocab = self.vocab.primitive
        processed_sentence = self.process([sentence], self.vocab.source)
        source_encodings, (last_encoder_state, last_encoder_cell) = TranxParser.encode(processed_sentence, [len(sentence)])
        source_encodings_attention_linear_layer = nn.Linear(256, 256, bias=False)
        decoder_initial_vector = F.tanh(self.decoder_cell_initializer_linear_layer(last_encoder_cell))
        h_t1 = decoder_initial_vector
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
            expanded_source_encodings_attention_linear_layer = \
                source_encodings_attention_linear_layer.expand(num_of_hypotheses,
                                                               source_encodings_attention_linear_layer.size(1),
                                                               source_encodings_attention_linear_layer.size(2))
            if t == 0:
                x = Variable(torch.cuda.FloatTensor(1, 128).zero_(), volatile=True)
            else:
                actions = [h.actions[-1] for h in hypotheses]
                action_embeddings = []
                for action in actions:
                    if action:
                        if isinstance(action, ApplyRuleAction):
                            action_embedding = self.apply_const_and_reduce_emb.weight[self.grammar.production_to_id[action.production]]
                        elif isinstance(action, ReduceAction):
                            action_embedding = self.apply_const_and_reduce_emb.weight[len(self.grammar)]
                        else:
                            action_embedding = self.primitives_emb.weight[self.vocab.primitive[action.token]]
                        action_embeddings.append(action_embedding)
                    else:
                        action_embeddings.append(Variable(torch.cuda.FloatTensor(128).zero_()))
                action_embeddings = torch.stack(action_embeddings)
                encoder_inputs = [action_embeddings]
                encoder_inputs.append(att_t1)

                frontier_fields = [h.frontier_field.field for h in hypotheses]
                frontier_field_embeddings = self.fields_emb(Variable(torch.cuda.FloatTensor([self.grammar.field_to_id[f] for f in frontier_fields])))
                encoder_inputs.append(frontier_field_embeddings)

                parent_created_times = [h.frontier_node.created_time for h in hypotheses]
                parent_states = torch.stack([hypotheses_states[h_id][parent_created_time][0]] for h_id, parent_created_time in enumerate(parent_created_times))
                parent_cells = torch.stack([hypotheses_states[h_id][parent_created_time][1] for h_id, parent_created_time in enumerate(parent_created_times)])
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

    def dot_product_attention(self, h_t, expanded_source_encodings, expanded_source_encodings_attention_linear_layer):
        attention_weight = F.softmax(
            torch.bmm(expanded_source_encodings_attention_linear_layer, h_t.unsqueeze(2)).squeeze(2), dim=-1)
        view = (attention_weight.size(0), 1, attention_weight.size(1))
        return torch.bmm(attention_weight.view(*view), expanded_source_encodings).squeeze(1)

    def step(self, x, h_t1, expanded_source_encodings, expanded_source_encodings_attention_linear_layer):
        h_t, cell_t = self.decoder(x, h_t1)
        context_t = self.dot_product_attention(h_t, expanded_source_encodings, expanded_source_encodings_attention_linear_layer)
        attention = F.tanh(self.attention_vector_lin(torch.cat([h_t, context_t], 1)))
        attention = self.dropout(attention)
        return (h_t, cell_t), attention

    def save(self, path):
        pass

    def load(self):
        pass

class PointerNet(nn.Module):
    def __init__(self, attention_vector_lin_size, source_encoding_size):
        super(PointerNet, self).__init__()
        self.source_encoding_linear = nn.Linear(source_encoding_size, attention_vector_lin_size, bias=False)

    def forward(self, source_encodings, attention_vector_lin):
        source_encodings = self.source_encoding_linear(source_encodings)
        attention = attention_vector_lin.permute(1, 0, 2).unsqueeze(3)
        weights = torch.matmul(source_encodings, attention).squeeze(3)
        weights = weights.permute(1, 0, 2)
        weights = F.softmax(weights, dim=-1)
        return weights
