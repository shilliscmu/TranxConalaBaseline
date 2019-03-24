"""
Get information about an action to be used when predicting actions
"""
from hypothesis import Hypothesis
from transitions import GenTokenAction


class ActionInfo(object):
    def __init__(self, action=None):
        self.t = 0
        self.parent_t = -1
        self.action = action
        self.frontier_prod = None
        self.frontier_field = None

        self.copy_from_src = False
        self.src_token_position = -1

    def __repr__(self, verbose=False):
        repr_str = '%s (t=%d, p_t=%d, frontier_field=%s)' % (repr(self.action),
                                                             self.t,
                                                             self.parent_t,
                                                             self.frontier_field.__repr__(
                                                                 True) if self.frontier_field else 'None')

        if verbose:
            verbose_repr = 'action_prob=%.4f, ' % self.action_prob
            if isinstance(self.action, GenTokenAction):
                verbose_repr += 'in_vocab=%s, ' \
                                'gen_copy_switch=%s, ' \
                                'p(gen)=%s, p(copy)=%s, ' \
                                'has_copy=%s, copy_pos=%s' % (self.in_vocab,
                                                              self.gen_copy_switch,
                                                              self.gen_token_prob, self.copy_token_prob,
                                                              self.copy_from_src, self.src_token_position)

            repr_str += '\n' + verbose_repr

        return repr_str

def get_action_infos(query, actions, force_copy=False):
    action_info_list = []
    prediction = Hypothesis()
    for t, action in enumerate(actions):
        action_info = ActionInfo(action)
        action_info.t = t
        if prediction.frontier_node:
            action_info.parent_t = prediction.frontier_node.created_time
            action_info.frontier_prod = prediction.frontier_node.production
            action_info.frontier_field = prediction.frontier_field.field

        if isinstance(action, GenTokenAction):
            try:
                token_source_index = query.index(str(action.token))
                action_info.copy_from_src = True
                action_info.src_token_position = token_source_index
            except ValueError:
                if force_copy:
                    raise ValueError('Can\'t copy input token %s' % action.token)
        prediction.apply_action(action)
        action_info_list.append(action_info)

    return action_info_list