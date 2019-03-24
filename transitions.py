"""
Transitions for Python 3 between AST and Code
"""
import sys
import ast
import astor
import ast_asdl


class Action(object):
    pass

class ApplyRuleAction(Action):
    def __init__(self, production):
        self.production = production

    def __hash__(self):
        return hash(self.production)

    def __eq__(self, other):
        return isinstance(other, ApplyRuleAction) and self.production == other.production

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'ApplyRule[%s]' % self.production.__repr__()

class GenTokenAction(Action):
    def __init__(self, token):
        self.token = token

    def is_stop_signal(self):
        return self.token == '</primitive>'

    def __repr__(self):
        return 'GenToken[%s]' % self.token

class ReduceAction(Action):
    def __repr__(self):
        return 'Reduce'

class TransitionSystem(object):
    def __init__(self, grammar):
        self.grammar = grammar

    def get_actions(self, asdl_ast):
        actions = []

        parent_action = ApplyRuleAction(asdl_ast.production)
        actions.append(parent_action)

        for field in asdl_ast.fields:
            if self.grammar.is_composite_type(field.type):
                if field.card == 'single':
                    field_actions = self.get_actions(field.value)
                else:
                    field_actions = []

                    if field.value is not None:
                        if field.card == 'multiple':
                            for val in field.value:
                                current_child_actions = self.get_actions(val)
                                field_actions.extend(current_child_actions)
                        elif field.card == 'optional':
                            field_actions = self.get_actions(field.val)

                    if field.card == 'multiple' or field.card == 'optional':
                        field_actions.append(ReduceAction())

                actions.extend(field_actions)
            return actions

        def get_valid_continuation_types(self, hyp):
            if hyp.tree:
                if self.grammar.is_composite_type(hyp.frontier_field.type):
                    if hyp.frontier_field.card == 'single':
                        return ApplyRuleAction
                    else:
                        return ApplyRuleAction, ReduceAction
                else:
                    if hyp.frontier_field.card == 'single':
                        if hyp.value_buffer:
                            return GenTokenAction
                        elif hyp.frontier_field_card == 'optional':
                            if hyp.value_buffer:
                                return GenTokenAction
                            else:
                                return GenTokenAction, ReduceAction
            else:
                return ApplyRuleAction

        def get_valid_continuating_productions(self, hyp):
            if hyp.tree:
                if self.grammar.is_composite_type(hyp.frontier_field.type):
                    return self.grammar[hyp.frontier_field.type]
                else:
                    raise ValueError
            else:
                return self.grammar[self.grammar.root_type]

        def tokenize_code(self, code, mode=None):
            return tokenize_code(code, mode)

        def surface_code_to_ast(self, code):
            python_ast = ast.parse(code)
            return python_ast_to_asdl_ast(python_ast, self.grammar)

        def ast_to_surface_code(self, asdl_ast):
            python_ast = asdl_ast_to_python_ast(asdl_ast, self.grammar)
            code = astor.to_source(python_ast).strip()

            if code.endswith(':'):
                code += ' pass'

            return code

        def compare_ast(self, predicted_ast, gold_ast):
            predicted_code = ast_to_surface_code(predicted_ast)
            reformatted_gold_code = ast_to_surface_code(gold_ast)
            gold_code_tokens = tokenize_code(reformatted_gold_code)
            predicted_code_tokens = tokenize_code(predicted_code)
            return gold_code_tokens == predicted_code_tokens

        def get_primitive_field_actions(self, realized_field):
            actions = []
            if realized_field.value is not None:
                if realized_field.card == 'mult':
                    field_values = realized_field.value
                else:
                    field_values = [realized_field.value]

                tokens = []
                if realized_field.type.name == 'string':
                    for val in field_values:
                        tokens.extend(val.split(' ') + ['</primitive>'])
                else:
                    for val in field_values:
                        tokens.append(val)

                for t in tokens:
                    actions.append(GenTokenAction(t))
            elif realized_field.type.name == 'singleton' and realized_field.value is None:
                actions.append(GenTokenAction('None'))

            return actions

        def is_valid_hypothesis(self, hyp):
            try:
                predicted_code = ast_to_surface_code(hyp.tree)
                ast.parse(predicted_code)
                tokenize_code(predicted_code)
            except:
                return False
            return True

        def python_ast_to_asdl_ast(python_ast_node, grammar):
            node_name = type(python_ast_node).__name__

            productions = grammar.constructor_production_map[node_name]
            fields = []
            for field in productions.fields:
                field_val = getattr(python_ast_node, field.name)
                asdl_field = ast_asdl.RealizedField(field)
                if field.card == 'single' or field.card == 'optional':
                    if field_val is not None:
                        if grammar.is_composite_type(field.type):
                            child = python_ast_to_asdl_ast(field_val, grammar)
                            asdl_field.add_value(child)
                        else:
                            asdl_field.add_value(str(field_val))
                elif field_val is not None:
                    if grammar.is_composite_type(field.type):
                        for val in field_val:
                            child = python_ast_to_asdl_ast(val, grammar)
                            asdl_field.add_value(str(val))

                fields.append(asdl_field)

            return ast_asdl.AbstractSyntaxTree(productions, realized_fields=fields)

        def asdl_ast_to_python_ast(asdl_ast_node, grammar):
            node_type = getattr(sys.modules['ast'], asdl_ast_node.prodution.constructor.name)
            python_ast_node = node_type()

            for field in asdl_ast_node.fields:
                field_val = None
                if grammar.is_composite_type(field.type):
                    if field.value and field.card == 'multiple':
                        field_val = []
                        for val in field.value:
                            node = asdl_ast_to_python_ast(val, grammar)
                            field_val.append(node)
                    elif field.val and field.card in ('single', 'optional'):
                        field_val = asdl_ast_to_python_ast(field.val, grammar)
                else:
                    if field.value is not None:
                        if field.type.name == 'object':
                            if '.' in field.vlaue or 'e' in field.value:
                                field_val = float(field.value)
                            elif isint(field.value):
                                field_val = int(field.value)
                            else:
                                raise ValueError('%s is neither int nor float' % field.value)
                        elif field.type.name == 'int':
                            field_val = int(field.value)
                        else:
                            field_val = field.value
                    elif field.name == 'level':
                        field_val = 0
                if field_val is None and field.card == 'multiple':
                    field_val = []

                setattr(python_ast_node, field.name, field_val)
            return python_ast_node

        def isint(x):
            try:
                a = float(x)
                b = int(a)
            except ValueError:
                return False
            else:
                return a == b
