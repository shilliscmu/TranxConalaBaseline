import json
import sys
import numpy as np
import torch
import torch.nn as nn

import evaluation
from asdl import ASDLGrammar
from preprocessor import PreProcessor
from transitions import TransitionSystem
from model import TranxParser
from evaluator import ConalaEvaluator

LR = 0.001
LR_DECAY = 0.5
DECAY_LR_AFTER_EPOCH = 15
BATCH_SIZE = 10
MAX_EPOCH = 50
GRAMMAR_FILE = "py3_asdl.simplified.txt"
PRIMITIVE_TYPES = ["identifier", "int", "string", "bytes", "object", "singleton"]
BEAM_SIZE = 15
PATIENCE = 5
SAVE_TO = "saved_models/conala"
SAVE_DECODE_TO = "saved_decode/decode_results.test.decode"
MAX_NUM_TRIAL = 5

def glorot_init(params):
    for p in params:
        if len(p.data.size()) > 1:
            nn.init.xavier_normal(p.data)

def batch_iter(data, batch_size, shuffle=False):
    index_arr = np.arange(len(data))
    if shuffle:
        np.random.shuffle(index_arr)

    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for batch_id in range(batch_num):
        batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
        batch_examples = [data[i] for i in batch_ids]
        batch_examples.sort(key=lambda e: -len(e.src_sent))

        yield batch_examples

def train(train_file_path):
    train_data, dev_data, vocab = PreProcessor.get_train_and_dev(train_file_path, GRAMMAR_FILE, PRIMITIVE_TYPES)

    grammar = ASDLGrammar.grammar_from_text(open(GRAMMAR_FILE).read(), PRIMITIVE_TYPES)
    transition_system = TransitionSystem(grammar)
    evaluator = ConalaEvaluator(transition_system)

    model = TranxParser(vocab, transition_system)
    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    glorot_init(model.parameters())

    epoch = train_iter = 0
    report_loss = report_examples = report_sup_att_loss = 0.
    history_dev_scores = []
    num_trial = patience = 0

    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in batch_iter(train_data, BATCH_SIZE):
            train_iter += 1
            optimizer.zero_grad()

            ret_val = model.forward(batch_examples)
            loss = -ret_val[0]

            loss_val = torch.sum(loss).data[0]
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            loss.backward()
            optimizer.step()

            if train_iter % 50 == 0:
                log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
                print(log_str)
                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin))

        model_file = SAVE_TO + '.iter%d.bin' % train_iter
        print('save model to [%s]' % model_file)
        model.save(model_file)

        # perform validation

        #if epoch % args.valid_every_epoch == 0:
        print('[Epoch %d] begin validation' % epoch)
        eval_start = time.time()
        eval_results = evaluation.evaluate(dev_data, model, evaluator, BEAM_SIZE,
                                           verbose=True)
        dev_score = eval_results[evaluator.default_metric]

        print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                            epoch, eval_results,
                            evaluator.default_metric,
                            dev_score,
                            time.time() - eval_start))

        is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
        history_dev_scores.append(dev_score)

        if DECAY_LR_AFTER_EPOCH and epoch > DECAY_LR_AFTER_EPOCH:
            lr = optimizer.param_groups[0]['lr'] * LR_DECAY
            print('decay learning rate to %f' % lr)

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if is_better:
            patience = 0
            model_file = SAVE_TO + '.bin'
            print('save the current model ..')
            print('save model to [%s]' % model_file)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), SAVE_TO + '.optim.bin')
        elif patience < PATIENCE and epoch >= DECAY_LR_AFTER_EPOCH:
            patience += 1
            print('hit patience %d' % patience)

        if epoch == MAX_EPOCH:
            print('reached max epoch, stop!')
            exit(0)

        if patience >= PATIENCE and epoch >= DECAY_LR_AFTER_EPOCH:
            num_trial += 1
            print('hit #%d trial' % num_trial)
            if num_trial == MAX_NUM_TRIAL:
                print('early stop!')
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * LR_DECAY
            print('load previously best model and decay learning rate to %f' % lr)

            # load model
            params = torch.load(SAVE_TO + '.bin', map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            model.to(device)

            # load optimizers
            """
            if args.reset_optimizer:
                print('reset optimizer')
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers')
                optimizer.load_state_dict(torch.load(SAVE_TO + '.optim.bin'))
            """

            print('reset optimizer')
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)


            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0

def test(test_file_path, model_path):
    test_data = PreProcessor.get_test(test_file_path, GRAMMAR_FILE, PRIMITIVE_TYPES)
    print('load model from [%s]' % model_path,)
    params = torch.load(model_path, map_location=lambda storage, loc: storage)
    transition_system = params['transition_system']

    parser = TranxParser.load(model_path=model_path)
    parser.eval()

    evaluator = ConalaEvaluator(transition_system)

    eval_results, decode_results = evaluation.evaluate(test_data, parser, evaluator, BEAM_SIZE,
                                                       verbose=True, return_decode_result=True)

    print(eval_results)
    pickle.dump(decode_results, open(SAVE_DECODE_TO, 'wb'))



if __name__ == '__main__':
    train_file_path = ""
    test_file_path = ""
    model_path = ""

    if sys.argv[1] == 'train':
        train(train_file_path)
    elif sys.argv[1] == 'test':
        test(test_file_path, model_path)
    else:
        raise RuntimeError('unknown mode')