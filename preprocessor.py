import ast
import json
import re
import torch

import astor
import numpy as np

from dataset_util import canonicalize_intent, canonicalize_code, tokenize_intent, decanonicalize_code
from transitions import TransitionSystem, GenTokenAction
from asdl import ASDLGrammar
from action_info import get_action_infos
from vocab import Vocab, VocabEntry

class PreProcessor(object):
    def __init__(self):
        self.token_to_index = {}

    def preprocess_dataset(self, file_path, transition_system):
        dataset = json.load(open(file_path))
        examples = []
        for i, example_json in enumerate(dataset):
            example_dict = PreProcessor.preprocess_example(example_json)

            # print(example_dict['canonical_snippet'] + "\n")
            python_ast = ast.parse(example_dict['canonical_snippet'])
            # print(ast.dump(python_ast))
            canonical_code = astor.to_source(python_ast).strip()
            tgt_ast = transition_system.python_ast_to_asdl_ast(python_ast, transition_system.grammar)
            tgt_actions = transition_system.get_actions(tgt_ast)
            # print("Intent tokens: ")
            # print(example_dict['intent_tokens'])
            # print("Target actions: ")
            # print(tgt_actions)
            tgt_action_infos = get_action_infos(example_dict['intent_tokens'], tgt_actions)

            example = Example(index=f'{i}-{example_json["question_id"]}',
                          sentence=example_dict['intent_tokens'],
                          tgt_actions=tgt_action_infos,
                          code=canonical_code,
                          ast=tgt_ast,
                          info=dict(example_dict=example_json,
                                    slot_map=example_dict['slot_map']))
            examples.append(example)

        return examples

    @staticmethod
    def preprocess_example(example_json):
        intent = example_json['intent']
        rewritten_intent = example_json['rewritten_intent']
        snippet = example_json['snippet']
        question_id = example_json['question_id']

        if rewritten_intent is None:
            rewritten_intent = intent

        canonical_intent, slot_map = canonicalize_intent(rewritten_intent)
        canonical_snippet = canonicalize_code(snippet, slot_map)
        intent_tokens = tokenize_intent(canonical_intent)
        decanonical_snippet = decanonicalize_code(canonical_snippet, slot_map)

        reconstructed_snippet = astor.to_source(ast.parse(snippet)).strip()
        reconstructed_decanonical_snippet = astor.to_source(ast.parse(decanonical_snippet)).strip()

        return {'canonical_intent': canonical_intent,
        'intent_tokens': intent_tokens,
        'slot_map': slot_map,
        'canonical_snippet': canonical_snippet}

    def get_train_and_dev(self, train_file_path, grammar_file, primitive_types):
        src_freq = 3
        code_freq = 3
        grammar = ASDLGrammar.grammar_from_text(open(grammar_file).read(), primitive_types)
        transition_system = TransitionSystem(grammar)
        train_examples = self.preprocess_dataset(train_file_path, transition_system)

        full_train_examples = train_examples[:]
        np.random.shuffle(train_examples)
        dev_examples = train_examples[:200]
        train_examples = train_examples[200:]

        src_vocab = VocabEntry.from_corpus([e.sentence for e in train_examples], size=5000,
                                       freq_cutoff=src_freq)
        primitive_tokens = [map(lambda a: a.action.token,
                                filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                            for e in train_examples]
        primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=5000, freq_cutoff=code_freq)

        # generate vocabulary for the code tokens!
        code_tokens = [transition_system.tokenize_code(e.code, mode='decoder') for e in train_examples]
        code_vocab = VocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=code_freq)

        vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab)

        return train_examples, dev_examples, vocab

    def get_test(self, test_file_path, grammar_file, primitive_types):
        grammar = ASDLGrammar.grammar_from_text(open(grammar_file).read(), primitive_types)
        transition_system = TransitionSystem(grammar)
        test_examples = PreProcessor.preprocess_dataset(test_file_path, transition_system)
        return test_examples


class Example(object):
    def __init__(self, index, sentence, tgt_actions, code, ast, info):
        self.index = index
        self.sentence = sentence
        self.tgt_actions = tgt_actions
        self.code = code
        self.ast = ast
        self.info = info