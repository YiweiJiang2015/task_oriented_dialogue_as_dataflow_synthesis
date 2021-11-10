"""
Author: Yiwei Jiang
Date: 12/09/2021
Functionality:
1. An object-wrapped lispress parser for SMCalFlow
2. Node.to_nested_dict() is the interface to tree_view.js in repo::smcal-web
"""

from enum import Enum
from typing import Union, List
import re, logging
from collections import Counter

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

LEFT_PAREN = "("
RIGHT_PAREN = ")"
ESCAPE = "\\"
DOUBLE_QUOTE = '"'
META = "^"
READER = "#"

# borrowed from lispress.py
# named args are given like `(fn :name1 arg1 :name2 arg2 ...)`
NAMED_ARG_PREFIX = ":"
# variables will be named `x0`, `x1`, etc., in the order they are introduced.
VAR_PREFIX = "x"

OperatorSet = {'&', '+', '-', '>', '<', '>=', '<=', '~=', '==', '?=', '?~=', '?>', '?<', '?<=', '?>='}

# Html Color Palette
BLACK_0 = 'black'
BLACK_1 = '#141414'
BROWN = 'brown'
RED = '#f54260'
BLUE = 'blue'
GREEN = 'green'
PURPLE = '#a134eb'
ORANGE = 'orange'
GOLD = 'gold'
PINK = 'pink'


class NodeTag(Enum):
    TypeHint = GREEN # 0 # ^
    KeyWord = BLUE # 1 # :
    SugarGet = ORANGE # 2 # #
    Op = RED # 3 # &, ?~=, ?=, ~=, >, <
    Call = BLACK_0 # 4 #
    Variable = GOLD # 5
    Value = PURPLE # 6
    Constraint = BROWN
    Misc = BLACK_1 # 100

    def __str__(self):
        return f'NodeTag: {self.name}'

    @classmethod
    def selected_name_iterator(cls):
        return [name for name in cls.__members__.keys() ] # if name not in ['Value']

    @classmethod
    def create_counter_by_name(cls):
        return {name: Counter() for name in cls.selected_name_iterator()}


class Node:
    def __init__(self, node_head: Union[List[str], str], state):
        self.node_head = node_head
        self.state = state
        self.edges = []
        self.tag = None # self.get_tag(state)

    def set_tag(self):

        if self.state == 'META_INIT':
            self.tag = NodeTag.TypeHint
        elif self.state == 'NAMED_ARG_PREFIX_INIT':
            self.tag = NodeTag.KeyWord
        elif self.state == 'READER_INIT':
            self.tag = NodeTag.SugarGet
        elif self.state == 'VAR_PREFIX_INIT':
            self.tag = NodeTag.Variable # if `x0` follows LEFT_PAREN, then this branch won't work.
        elif self.state == 'DOUBLE_QUOTE_INIT':
            self.tag = NodeTag.Value
        else:
            if not self.head_is_list:
                if self.node_head in OperatorSet:
                    self.tag = NodeTag.Op
                elif self.node_head.startswith(VAR_PREFIX):
                    self.tag = NodeTag.Variable
                elif self.node_head.startswith(':'):
                    self.tag = NodeTag.KeyWord
                elif 'Constraint' in self.node_head:
                    self.tag = NodeTag.Constraint
                elif len(self.edges) == 0 or re.match(r'(\d+L)|(true)', self.node_head):
                    self.tag = NodeTag.Value
                else:
                    self.tag = NodeTag.Call
            else:
                if self.head_list_contains_value:
                    self.tag = NodeTag.Value
                elif self.head_list_contains_op:
                    self.tag = NodeTag.Op

                else:
                    # todo I cannot differentiate Call from Struct (e.g. ^(Date) EmptyStructConstraint?). So, this might be not precise.
                    self.tag = NodeTag.Call

        # return tag

    def add_edge(self, edge):
        self.edges.append(edge)

    # def parse_lispress_string(self, lisp_string: str):
    #     return

    def walk(self, visitor):
        visitor.node(self)

        for edge in self.edges:
            edge.walk(visitor)

    def to_nested_dict(self) -> dict:
        """
        Conform to the jstree data format.
        https://www.jstree.com/docs/json/
        """
        nested_dict = {
            "text": ' '.join(self.node_head) if self.head_is_list else self.node_head,
            'icon': False,
            'state': {'opened': True, 'selected': False},
            'children': [edge.pop_end_point_to_nested_dict() for edge in self.edges],
            'li_attr': {'style': f'color:{self.tag.value}'}
        }
        return nested_dict

    def __str__(self):
        return f'Node_head: `{self.node_head}`. {self.num_children} children. {self.tag}'

    @property
    def head_list_contains_value(self):
        # todo normally, a node head constituted by a list is a sugar expression, right?
        # todo can we find any exception?
        # also if the len(node_head) is larger than 3. There may be something wrong...
        # "'HourMinuteMilitary', '11L', '0L'" is an legitimate example of 3 elements
        if len(self.node_head) > 3:
            logger.warning(f'The node {self.node_head} has a list head with {len(self.node_head)} elements.')
        last_span = self.node_head[-1]
        last_ch = last_span[-1]
        condition_a = last_ch in {'L', '"', ']'}.union({str(d) for d in range(10)}) and not last_span.startswith('x')
        condition_b = last_span.lower() == 'true' or last_span.lower() == 'false'
        if condition_a or condition_b: # use reg?
            return True
        else:
            return False

    @property
    def head_list_contains_op(self):
        last_span = self.node_head[-1]
        if last_span in OperatorSet:
            return True
        else:
            return False

    @property
    def head_is_list(self):
        return isinstance(self.node_head, list)

    @property
    def num_children(self):
        return len(self.edges)


class Edge:
    def __init__(self, edge_start_point: Node, edge_end_point: Node):
        self.start_point = edge_start_point
        self.end_point = edge_end_point

    # def __repr__(self):
    #     return self.to_eamr(multiline=False)

    # This method expects a Visitor object as argument
    def walk(self, visitor):
        visitor.edge(self)
        self.end_point.walk(visitor)

    def pop_end_point_to_nested_dict(self):
        return self.end_point.to_nested_dict()

    def __str__(self):
        return f'End [{self.end_point}]; Start [{self.start_point}]'


class Visitor:
    'Interface'

    def node(self, node):
        return

    def edge(self, edge):
        return

    # def span(self, span):
    #     return


class SexpParser:
    """
    A bottom-up parser.
    """

    def __init__(self, text):
        self.buffer = MyBuffer(text)

    def parse(self) -> Node:
        node = self.parse_node()
        self.buffer.end_of_stream()
        return node

    def parse_node(self):
        node_head = None
        state = None
        if self.buffer.skip_then_peek() == LEFT_PAREN:
            self.buffer.accept(LEFT_PAREN)
            state = "PAREN_INIT"
        elif self.buffer.skip_then_peek() == READER:
            state = "READER_INIT"
        elif self.buffer.skip_then_peek() == META:
            state = "META_INIT"
        elif self.buffer.skip_then_peek() == NAMED_ARG_PREFIX:
            state = "NAMED_ARG_PREFIX_INIT"
        elif self.buffer.skip_then_peek() == VAR_PREFIX:
            state = "VAR_PREFIX_INIT"
        elif self.buffer.skip_then_peek() == DOUBLE_QUOTE:
            state = "DOUBLE_QUOTE_INIT"
        else:
            pass

        if self.buffer.peek() != LEFT_PAREN and self.buffer.peek() != RIGHT_PAREN:
            node_head = self.parse_node_head()

        node = Node(node_head, state)
        # accept edges
        c = self.buffer.skip_then_peek() # Storing the peeked letter into a variable will make the loop condition cleaner and more efficient.
        while c == LEFT_PAREN or c.isnumeric() or c == READER or c == NAMED_ARG_PREFIX \
                or c == VAR_PREFIX or c == META or c == DOUBLE_QUOTE or c != RIGHT_PAREN:
            node.add_edge(self.parse_edge(node))
            if self.buffer.is_eoi():
                # If a string ends with `#(PlaceFeature "FullBar")` or `??` todo I don't recall any other situations now,
                # the buffer will exhaust the last RIGHT_PAREN and eventually points to End_of_String.
                # Thus, we need to jump out of the loop. Otherwise, it will cause IndexError within buffer.
                break
            if state == 'READER_INIT' or state == 'META_INIT' or state == 'NAMED_ARG_PREFIX_INIT':
                # Operators (i.e., `^`, `#`, `:`) only accept one child
                break
                # c = self.buffer.peek()
            else:
                c = self.buffer.skip_then_peek()
        # end of node
        if state == "PAREN_INIT":
            self.buffer.accept(RIGHT_PAREN)
        node.set_tag()
        return node

    def parse_edge(self, edge_start_point):
        edge_end_point = self.parse_node()
        return Edge(edge_start_point, edge_end_point)

    def parse_node_head(self):
        return self.buffer.find_node_head()

    def __str__(self):
        return f'Parser buffer state-{str(self.buffer)}'


class SexpParseError(Exception):
    pass


class MyBuffer:

    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.length = len(self.text)

    def is_eoi(self) -> bool:
        return self.pos == self.length

    def peek(self) -> str:
        """Read ahead of the character of current position from self.text"""
        return self.text[self.pos]

    def next_char(self):
        # pylint: disable=used-before-assignment
        cn = self.text[self.pos]
        self.pos += 1
        return cn

    def skip_whitespace(self):
        while (not self.is_eoi()) and self.peek().isspace():
            self.next_char()

    def skip_then_peek(self) -> str:
        self.skip_whitespace()
        return self.peek()

    def find_meta_node_head(self) -> str:
        meta_node_head = self.find_node_head()
        if isinstance(meta_node_head, list):
            return ' '.join(meta_node_head)
        elif isinstance(meta_node_head, str):  # str
            return meta_node_head
        else:
            raise SexpParseError(f"Wrong characters after `^` in '{self.text[:self.pos]}*{self.text[self.pos:]}")

    def find_meta_span(self):
        """
        Find the text spans after `^`. There are 3 varieties found in current data:
        - Single word: ^(Date), ^Date, ^Recipient
        - Consecutive words: ^(Constraint Event), ^(CalflowIntension Event)
        - Nested: ^(Constraint (List Attendee)), ^(Constraint (CalflowIntension Event)), ^((Constraint (List Attendee)) Unit), ^((Constraint ShowAsStatus) Unit)
        """
        meta_span = ""
        meta_state = None  # 0: meta followed by LEFT_PAREN, 1: meta followed directly by letters
        if self.peek() == LEFT_PAREN:
            meta_state = 0
            self.accept(LEFT_PAREN)
            meta_span += self.find_meta_node_head()  # todo I really want to avoid this func
            while self.peek() != RIGHT_PAREN: #== LEFT_PAREN:
                meta_span += (' ' + self.find_meta_span()) # handles nested meta spans like ^(Constraint (CalflowIntension Event))
            self.accept(RIGHT_PAREN)
        else:
            meta_state = 1
            meta_span += self.find_meta_node_head()  # I really want to avoid this func
        return f"({meta_span.lstrip()})" if meta_state == 0 else meta_span

    def find_reader_span(self):
        """ Find the text span after `#` like `(Number 3)` or `(String "meeting with the lecture")` """
        if self.peek() != LEFT_PAREN:
            raise SexpParseError(f"Wrong characters after `^` in '{self.text[:self.pos]}*{self.text[self.pos:]}")
        self.accept(LEFT_PAREN)
        reader_span = ' '.join(self.find_node_head())
        if self.peek() != RIGHT_PAREN:
            raise SexpParseError(f"Wrong characters after `^` in '{self.text[:self.pos]}*{self.text[self.pos:]}")

        return reader_span

    def find_consecutive_span(self):
        """This func is mainly borrowed from dataflow.core.sexp.py-read_list()"""
        c = self.text[self.pos]
        if c == DOUBLE_QUOTE:
            self.accept(DOUBLE_QUOTE)
            out_str = ""
            while self.peek() != DOUBLE_QUOTE:
                c_string = self.next_char()
                out_str += c_string
                if c_string == ESCAPE:
                    out_str += self.next_char()
            self.next_char()
            return f'"{out_str}"'
        elif c == META:
            self.accept(META)
            if self.peek() == LEFT_PAREN:
                meta_span = self.find_meta_span() # see notes below this func
                expr_span = self.find_consecutive_span() #
                if expr_span != '':
                    return [META + meta_span, expr_span]
                else:
                    return META + meta_span
            else:
                return META + self.find_meta_span()
        else:
            out_inner = ""
            # if c != "\\":
            #     out_inner += c

            # TODO: is there a better loop idiom here?
            if not self.is_eoi():
                next_c = self.skip_then_peek()  # peek
                escaped = c == ESCAPE
                while (not self.is_eoi()) and (
                        escaped or not _is_beginning_control_char(next_c)
                ):
                    if (not escaped) and next_c == ESCAPE:
                        self.next_char()
                        escaped = True
                    else:
                        out_inner += self.next_char()
                        escaped = False
                    if not self.is_eoi():
                        next_c = self.peek()
            return out_inner

    def find_node_head(self):
        out_head = []
        if self.skip_then_peek() == READER:
            self.accept(READER)
            return READER
        if self.skip_then_peek() == NAMED_ARG_PREFIX:
            self.accept(NAMED_ARG_PREFIX)
            return NAMED_ARG_PREFIX + self.find_consecutive_span()
        if self.skip_then_peek() == DOUBLE_QUOTE:
            return self.find_consecutive_span()
        if self.skip_then_peek() == VAR_PREFIX:
            return self.find_consecutive_span()
        while self.skip_then_peek() != RIGHT_PAREN:
            c = self.skip_then_peek()
            if c == LEFT_PAREN or c == NAMED_ARG_PREFIX or c == READER or c == DOUBLE_QUOTE or c == VAR_PREFIX:
                break
            found_span = self.find_consecutive_span()
            if isinstance(found_span, list):
                out_head.extend(found_span)
            else:
                out_head.append(found_span)
        return out_head[0] if len(out_head) == 1 else out_head

    def accept(self, string):
        """Check if the string passed in matches indexed raw text"""
        self.skip_whitespace()  # self.pos ++ if current pos points to a whitespace.
        end = self.pos + len(string)
        if end <= self.length and string == self.text[self.pos:end]:
            self.pos += len(string)
            return
        else:
            raise SexpParseError(f"failed to accept '{string}' in '{self.text[:self.pos]}*{self.text[self.pos:]}'")

    def end_of_stream(self):
        """
        When the buffer is exhausted, the cursor (self.pos) shall point to the end of string,
        namely self.pos==self.length.
        """
        self.skip_whitespace()

        if self.pos < self.length:
            raise SexpParseError("invalid end '^{}'".format(self.text[self.pos:]))

    def __str__(self):
        return f"Buffer state--length: {self.length}, pos: {self.pos}, caret: {self.text[self.pos]}"


def _is_beginning_control_char(nextC):
    return (
            nextC.isspace()
            or nextC == LEFT_PAREN
            or nextC == RIGHT_PAREN
            or nextC == DOUBLE_QUOTE
            or nextC == READER
            or nextC == META
            or nextC == NAMED_ARG_PREFIX
    )


if __name__ == "__main__":
    test_string = r"""
(Yield :output (:start (FindNumNextEvent :constraint (Constraint[Event] :subject (?~= #(String "staff meeting"))) :number #(Number 1))))
    """  # ^(Constraint (CalflowIntension Event))
    parser = SexpParser(test_string)

    node = parser.parse()
    nested_dict = node.to_nested_dict()
    print()
    print(node)
