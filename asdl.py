"""
ASDL grammar class and production class
"""
import re
from collections import OrderedDict
from itertools import chain


class ASDLGrammar(object):
    def __init__(self, productions):
        # dict of productions by head types
        self.productions = OrderedDict()
        # map of productions by constructor
        self.constructor_production_map = {}
        for p in productions:
            if p.type not in self.productions:
                self.productions[p.type] = []
            self.productions[p.type].append(p)
            self.constructor_production_map[p.constructor.name] = p
        
        self.root_type = productions[0].type
        # num of constructors
        self.size = sum(len(head) for head in self.productions.values())

        self.productions = self.get_productions()
        self.types = self.get_types()
        self.fields = self.get_fields()
        self.composite_types = self.get_composite_types()
        self.primitive_types = self.get_primitives_types()
        
        # mappings of entities to ids
        self.production_to_id = {p: index for index, p in enumerate(self.productions)}
        self.type_to_id = {t: index for index, t in enumerate(self.types)}
        self.field_to_id = {f: index for index, f in enumerate(self.fields)}

        self.id_to_production = {index: p for index, p in enumerate(self.productions)}
        self.id_to_type = {index: type for index, type in enumerate(self.types)}
        self.id_to_field = {index: field for index, field in enumerate(self.fields)}

    def __len__(self):
        return self.size

    def __getitem__(self, datum):
        if isinstance(datum, str):
            return self.productions[ASDLType(datum)]
        else:
            return self.productions[datum]

    # get a list of all production values
    def get_productions(self):
        productions = sorted(chain.from_iterable(self.productions.values()), key=lambda x: repr(x))
        return productions

    def get_productions_by_constructor_name(self, constructor_name):
        return self.constructor_production_map[constructor_name]

    # get a set of all types used in productions
    def get_types(self):
        types = set()
        for p in self.productions:
            types.add(p.type)
            types.update(map(lambda x: x.type, p.constructor.fields))
        types = sorted(types, key=lambda x: x.name)
        return types

    # get a set of all fields used in productions
    def get_fields(self):
        fields = set()
        for p in self.productions:
            fields.update(p.constructor.fields)
        fields = sorted(fields, key=lambda x: (x.name, x.type.name, x.card))
        return fields

    # filter out the primitive types from all the types
    def get_primitives_types(self):
        primitive_types = filter(lambda x: isinstance(x, ASDLPrimitiveType), self.types)
        return primitive_types

    def get_composite_types(self):
        composite_types = filter(lambda x: isinstance(x, ASDLCompositeType), self.types)
        return composite_types

    def is_composite_type(self, asdl_type):
        return asdl_type in self.composite_types

    def is_primitive_type(self, asdl_type):
        return asdl_type in self.primitive_types

    @staticmethod
    def grammar_from_text(text, primitive_types):
        def get_field_from_text(text):
            text = text.strip().split(' ')
            type = text[0].strip()
            name = text[1].strip()
            card = 'single'
            if type[-1] == '*':
                type = type[:-1]
                card = 'multiple'
            elif type[-1] == '?':
                type = type[:-1]
                card = 'optional'

            if type in primitive_types:
                return Field(name, ASDLPrimitiveType(type), card=card)
            else:
                return Field(name, ASDLCompositeType(type), card=card)

        def get_constructor_from_text(text):
            text = text.strip()
            fields = None
            if '(' in text:
                name = text[:text.find('(')]
                field_blocks = text[text.find('(') + 1:text.find(')')].split(',')
                fields = map(get_field_from_text, field_blocks)
            else:
                name = text

            if name == '':
                name = None

            return ASDLConstructor(name, fields)

        lines = re.sub(re.compile("#.*"), "", text)
        lines = '\n'.join(filter(lambda x: x, lines.split('\n')))
        lines = lines.split('\n')
        lines = list(map(lambda l: l.strip(), lines))
        lines = list(filter(lambda l: l, lines))
        line_num = 0

        # first line is primitive type
        primitive_types = list(map(lambda x: x.strip(), lines[line_num].split(',')))
        line_num += 1

        all_productions = list()

        while True:
            type_block = lines[line_num]
            type_name = type_block[:type_block.find('=')].strip()
            constructors_blocks = type_block[type_block.find('=') + 1:].split('|')
            index = line_num + 1
            while index < len(lines) and lines[index].strip().startswith('|'):
                t = lines[index].strip()
                cont_constructors_blocks = t[1:].split('|')
                constructors_blocks.extend(cont_constructors_blocks)
                index += 1

            constructors_blocks = filter(lambda x: x and x.strip(), constructors_blocks)

            # get type name
            if type_name in primitive_types:
                new_type = ASDLPrimitiveType(type_name)
            else:
                new_type = ASDLCompositeType(type_name)

            constructors = map(get_constructor_from_text, constructors_blocks)

            productions = list(map(lambda x: ASDLProduction(new_type, x), constructors))
            all_productions.extend(productions)

            line_num = index
            if line_num == len(lines):
                break

        return ASDLGrammar(all_productions)

class ASDLProduction(object):
    def __init__(self, type, constructor):
        self.type = type
        self.constructor = constructor

    def __getitem__(self, field):
        return self.constructor[field]

    def __hash__(self):
        return hash(self.type) ^ hash(self.constructor)

    def __eq__(self, other):
        return isinstance(other, ASDLProduction) and self.type == other.type and self.constructor == other.constructor

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '%s -> %s' % (self.type.__repr__(plain=True), self.constructor.__repr__(plain=True))

    def fields(self):
        return self.constructor.fields

class ASDLConstructor(object):
    def __init__(self, name, fields=None):
        self.name = name
        self.fields = []
        if fields:
            self.fields = list(fields)

    def __getitem__(self, field_name):
        for field in self.fields:
            if field.name == field.name:
                return field

        raise KeyError

    def __hash__(self):
        h = hash(self.name)
        for field in self.fields:
            h ^= hash(field)

        return h

    def __eq__(self, other):
        return isinstance(other, ASDLConstructor) and self.name == other.name and self.fields == other.fields

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, plain=False):
        plain_repr = '%s(%s)' % (self.name, ', '.join(f.__repr__(plain=True) for f in self.fields))

        if plain:
            return plain_repr
        else:
            return 'Constructor(%s)' % plain_repr

class Field(object):
    def __init__(self, name, type, card):
        self.name = name
        self.type = type

        assert card in ['single', 'optional', 'multiple']
        self.card = card

    def __hash__(self):
        h = hash(self.name) ^ hash(self.type)
        h ^= hash(self.card)

        return h

    def __eq__(self, other):
        return isinstance(other, Field) and self.name == other.name and self.type == other.type and self.card == other.card

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, plain=False):
        if self.card == 'single':
            card_repr = ''
        elif self.card =='optional':
            card_repr = '?'
        else:
            card_repr = '*'

        plain_repr = '%s%s %s' % (self.type.__repr__(plain=True), card_repr, self.name)

        if plain:
            return plain_repr
        else:
            return 'Field(%)' % plain_repr

class ASDLType(object):
    def __init__(self, type_name):
        self.name = type_name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, ASDLType) and self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, plain=False):
        plain_repr = self.name
        if plain:
            return plain_repr
        else:
            return '%s(%s)' % (self.__class__.__name__, plain_repr)

# just used to differentiate the different different types using isinstance

class ASDLCompositeType(ASDLType):
    pass

class ASDLPrimitiveType(ASDLType):
    pass

