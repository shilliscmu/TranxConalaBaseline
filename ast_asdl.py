"""
Constructing, copying, and printing abstract syntax trees.
"""
from io import StringIO
from asdl import Field, ASDLCompositeType


class AbstractSyntaxTree(object):
    def __init__(self, production, realized_fields=None):
        self.production = production

        self.fields = []
        self.parent_field = None
        self.created_time = 0
        if realized_fields:
            for field in realized_fields:
                self.add_child(field)
        else:
            for field in self.production.fields():
                self.add_child(RealizedField(field))

    def __getitem__(self, field_name):
        for field in self.fields:
            if field.name == field_name:
                return field
        raise KeyError

    def __hash__(self):
        code = hash(self.production)
        for field in self.fields:
            code = code + 37 * hash(field)

        return code

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.created_time != other.created_time:
            return False
        if self.production != other.production:
            return False
        if len(self.fields) != len(other.fields):
            return False
        for i in range(len(self.fields)):
            if self.fields[i] != other.fields[i]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(self.production)

    def add_child(self, realized_field):
        self.fields.append(realized_field)
        realized_field.parent_node = self

    def copy(self):
        copy = AbstractSyntaxTree(self.production)
        copy.created_time = self.created_time
        for i, old_field in enumerate(self.fields):
            new_field = copy.fields[i]
            new_field.not_single_cardinality_finished = old_field.not_single_cardinality_finished
            if isinstance(old_field.type, ASDLCompositeType):
                for value in old_field.as_value_list:
                    new_field.add_value(value.copy())
            else:
                for value in old_field.as_value_list:
                    new_field.add_value(value)
        return copy

    def to_string(self, sb=None):
        is_root = False
        if sb is None:
            is_root = True
            sb = StringIO()
        sb.write('(')
        sb.write(self.production.constructor.name)

        for field in self.fields:
            sb.write(' ')
            sb.write('(')
            sb.write(field.type.name)
            if field.card == 'single':
                card = ''
            elif field.card == 'optional':
                card = '?'
            else:
                card = '*'
            sb.write(card)
            sb.write('-')
            sb.write(field.name)

            if field.value is not None:
                for val_node in field.as_value_list():
                    sb.write(' ')
                    if isinstance(field.type, ASDLCompositeType):
                        val_node.to_string(sb)
                    else:
                        sb.write(str(val_node).replace(' ', '-SPACE-'))
            sb.write(')')
        sb.write(')')

        if is_root:
            return sb.getvalue()

    def size(self):
        node_count = 1
        for field in self.fields:
            for val in field.as_value_list():
                if isinstance(val, AbstractSyntaxTree):
                    node_count += val.size
                else:
                    node_count = node_count + 1
        return node_count


class RealizedField(Field):
    def __init__(self, field, value=None, parent=None):
        super(RealizedField, self).__init__(field.name, field.type, field.card)
        self.parent_node = None
        self.field = field
        if self.card == 'multiple':
            self.value = []
            if value is not None:
                for child_node in value:
                    self.add_value(child_node)
        else:
            self.value = None
            if value is not None:
                self.add_value(value)

        self.not_single_cardinality_finished = False

    def __eq__(self, other):
        if super(RealizedField, self).__eq__(other):
            if type(other) == Field:
                return True
            if self.value == other.value:
                return True
            else:
                return False
        else:
            return False

    def add_value(self, value):
        if isinstance(value, AbstractSyntaxTree):
            value.parent_field = self
        if self.card == 'multiple':
            self.value.append(value)
        else:
            self.value = value

    @property
    def as_value_list(self):
        if self.card == 'multiple':
            return self.value
        elif self.value is not None:
            return [self.value]
        else:
            return []

    def is_finished(self):
        if self.card == 'single':
            if self.value is None:
                return False
            else:
                return True
        elif self.card == 'optional' and self.value is not None:
                return True
        else:
            if self.not_single_cardinality_finished:
                return True
            else:
                return False

    def set_finished(self):
        self.not_single_cardinality_finished = True
