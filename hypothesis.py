"""
Maintains and manipulates a hypothesis tree; used for decoding.
"""
from ast_asdl import AbstractSyntaxTree
from transitions import ApplyRuleAction, ReduceAction, GenTokenAction
from asdl import ASDLCompositeType


class Hypothesis(object):
    def __init__(self):
        self.tree = None
        self.actions = []
        self.score = 0.
        self.frontier_node = None
        self.frontier_field = None
        self.value_buffer = []
        self.t = 0

    def apply_action(self, action):
        # self.frontier_field is none, so it defaults to reducing
        # print(self.frontier_field.type)
        if self.tree is None:
            assert isinstance(action, ApplyRuleAction), 'Invalid action [%s], only ApplyRule action is valid ' \
                                                        'at the beginning of decoding'

            self.tree = AbstractSyntaxTree(action.production)
            self.update_frontier()
        elif self.frontier_node:
            # print(action.__repr__())
            # print(self.frontier_node)
            if isinstance(self.frontier_field.type, ASDLCompositeType):
                if isinstance(action, ApplyRuleAction):
                    field_val = AbstractSyntaxTree(action.production)
                    field_val.create_time = self.t
                    self.frontier_field.add_value(field_val)
                    self.update_frontier()
                elif isinstance(action, ReduceAction):
                    assert self.frontier_field.card in ('optional', 'multiple'), 'Reduce action can only be ' \
                                                                                 'applied on field with multiple ' \
                                                                                 'cardinality'
                    self.frontier_field.set_finished()
                    self.update_frontier()
                else:
                    raise ValueError('Can\'t do %s on field %s' % (action, self.frontier_field))
            else:
                if isinstance(action, GenTokenAction):
                    end_prim = False
                    if self.frontier_field.type.name == 'string':
                        if action.is_stop_signal():
                            self.frontier_field.add_value(' '.join(self.value_buffer))
                            self.value_buffer = []
                            end_prim = True
                        else:
                            self.value_buffer.append(action.token)
                    else:
                        self.frontier_field.add_value(action.token)
                        end_prim = True
                    if end_prim and self.frontier_field.card in ('single', 'optional'):
                        self.frontier_field.set_finished()
                        self.update_frontier()
                elif isinstance(action, ReduceAction):
                    # print(self.frontier_field.card)
                    assert self.frontier_field.card in ('optional', 'multiple'), 'Reduce action can only be ' \
                                                                                        'applied on field with multiple ' \
                                                                                        'cardinality'
                    self.frontier_field.set_finished()
                    self.update_frontier()
                else:
                    raise ValueError('With a primitive field, you can only generate or reduce.')

        self.t += 1
        self.actions.append(action)

    def update_frontier(self):
        def find_frontier_node_and_field(tree_node):
            if tree_node:
                for field in tree_node.fields:
                    if isinstance(field.type, ASDLCompositeType) and field.value:
                        if field.card in ('single', 'optional'):
                            vals = [field.value]
                        else:
                            vals = field.value
                        for child in vals:
                            result = find_frontier_node_and_field(child)
                            if result:
                                return result
                    if not field.is_finished():
                        return tree_node, field
                return None
            else:
                return None

        frontier = find_frontier_node_and_field(self.tree)
        if frontier:
            self.frontier_node, self.frontier_field = frontier
        else:
            self.frontier_node, self.frontier_field = None, None

    def clone_and_apply_action(self, action):
        new_hypothesis = self.copy()
        new_hypothesis.apply_action(action)
        return new_hypothesis

    def copy(self):
        new_hypothesis = Hypothesis()
        if self.tree:
            new_hypothesis.tree = self.tree.copy()

        new_hypothesis.actions = list(self.actions)
        new_hypothesis.score = self.score
        new_hypothesis.value_buffer = list(self.value_buffer)
        new_hypothesis.t = self.t
        new_hypothesis.update_frontier()
        return new_hypothesis

    def is_completed(self):
        return self.tree and self.frontier_field is None
