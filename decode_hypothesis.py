from hypothesis import Hypothesis


class DecodeHypothesis(Hypothesis):
    def __init__(self):
        super(DecodeHypothesis, self).__init__()
        self.action_infos = []
        self.code = None

    def clone_and_apply_action_info(self, action_info):
        action = action_info.action
        new_hypothesis = self.clone_and_apply_action(action)
        new_hypothesis.action_infos.append(action_info)
        return new_hypothesis

    def copy(self):
        new_hypothesis = DecodeHypothesis()
        if self.tree:
            new_hypothesis.tree = self.tree.copy()

        new_hypothesis.actions = list(self.actions)
        new_hypothesis.action_infos = list(self.action_infos)
        new_hypothesis.score = self.score
        new_hypothesis.value_buffer = list(self.value_buffer)
        new_hypothesis.t = self.t
        new_hypothesis.code = self.code
        new_hypothesis.update_frontier()
        return new_hypothesis