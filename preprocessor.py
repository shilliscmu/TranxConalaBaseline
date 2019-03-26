import json
import re
import numpy as np

from transitions import TransitionSystem
from asdl import ASDLGrammar
from dataset_util import *
from action_info import get_action_infos
from vocab import Vocab, VocabEntry

class Example(object):
    def __init__(self, src_sent, tgt_actions, tgt_code, tgt_ast, idx=0, meta=None):
        self.src_sent = src_sent
        self.tgt_code = tgt_code
        self.tgt_ast = tgt_ast
        self.tgt_actions = tgt_actions

        self.idx = idx
        self.meta = meta

class PreProcessor(object):
    def preprocess_dataset(file_path):
        dataset = json.load(open(file_path))
        examples = []
        for i, example_json in enumerate(dataset):
            example_dict = PreProcessor.preprocess_example(example_json)

            python_ast = ast.parse(example_dict['canonical_snippet'])
            canonical_code = astor.to_source(python_ast).strip()
            tgt_ast = python_ast_to_asdl_ast(python_ast, transition_system.grammar)
            tgt_actions = transition_system.get_actions(tgt_ast)
            tgt_action_infos = get_action_infos(example_dict['intent_tokens'], tgt_actions)

            example = Example(idx=f'{i}-{example_json["question_id"]}',
                          src_sent=example_dict['intent_tokens'],
                          tgt_actions=tgt_action_infos,
                          tgt_code=canonical_code,
                          tgt_ast=tgt_ast,
                          meta=dict(example_dict=example_json,
                                    slot_map=example_dict['slot_map']))

            examples.append(example)
        return examples

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

    @staticmethod
    def get_train_and_dev(train_file_path, grammar_file, premitive_types):
        grammar = ASDLGrammar.grammar_from_text(open(grammar_file).read(), premitive_types)
        transition_system = TransitionSystem(grammar)
        train_examples = PreProcessor.preprocess_dataset(train_file_path)

        full_train_examples = train_examples[:]
        np.random.shuffle(train_examples)
        dev_examples = train_examples[:200]
        train_examples = train_examples[200:]

        src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_examples], size=5000,
                                       freq_cutoff=src_freq)
        primitive_tokens = [map(lambda a: a.action.token,
                                filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                            for e in train_examples]
        primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=5000, freq_cutoff=code_freq)

        code_tokens = [transition_system.tokenize_code(e.tgt_code, mode='decoder') for e in train_examples]
        code_vocab = VocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=code_freq)

        vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab)

        return train_examples, dev_examples, vocab

    @staticmethod
    def get_test(test_file_path):
        test_examples = PreProcessor.preprocess_dataset(test_file_path)
        return test_examples
