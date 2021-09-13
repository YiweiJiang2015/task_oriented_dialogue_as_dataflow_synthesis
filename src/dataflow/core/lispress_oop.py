from enum import Enum
from typing import Union, List
from dataflow.core.lispress import (
    META_CHAR,
    VALUE_CHAR,
    # lispress_to_type_name,
    # parse_lispress,
)
LEFT_PAREN = "("
RIGHT_PAREN = ")"
ESCAPE = "\\"
DOUBLE_QUOTE = '"'
META = "^"
READER = "#"

class NodeType:
    pass


class Node:
    def __init__(self, node_head: Union[List[str], str]):
        self.node_head = node_head
        self.edges = []
        self.type = None

    def add_edge(self, edge):
        self.edges.append(edge)

    def parse_lispress_string(self, lisp_string: str):
        return

    def walk(self, visitor):
        visitor.node(self)

        for edge in self.edges:
            edge.walk(visitor)

    def to_nested_dict(self) -> dict:
        """
        Comform to the jstree data format.
        https://www.jstree.com/docs/json/
        """
        nested_dict = {
            "text": ' '.join(self.node_head) if self.head_is_list else self.node_head,
            'icon': False,
            'state': {'opened': True, 'selected': False},
            'children': [edge.pop_end_point_to_nested_dict() for edge in self.edges]
        }
        return nested_dict

    def __str__(self):
        return f'Node_head: {self.node_head}, {self.num_children} children'

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
        return f'Start [{self.start_point}]; End [{self.end_point}]'

class Visitor:
    'Interface'
    def node(self, node):
        return

    def edge(self, edge):
        return

    def span(self, span):
        return


class SexpParser:
    """
    A bottom-up parser.
    """
    def __init__(self, text):
        self.buffer = MyBuffer(text)

    def parse(self):
        node = self.parse_node()
        self.buffer.end_of_stream()
        return node

    def parse_node(self):
        node_head = None
        state = None
        if self.buffer.skip_then_peek() == LEFT_PAREN:
            self.buffer.accept(LEFT_PAREN)
            state = "PAREN_INIT"
        else:
            pass # todo

        if self.buffer.peek() != LEFT_PAREN and self.buffer.peek() != RIGHT_PAREN:
            node_head = self.find_node_head()

        node = Node(node_head)
        # accept edges
        while self.buffer.skip_then_peek() == LEFT_PAREN or self.buffer.skip_then_peek().isnumeric():
            node.add_edge(self.parse_edge(node))
        # end of node
        if state == "PAREN_INIT":
            self.buffer.accept(RIGHT_PAREN)
        return node

    def parse_edge(self, edge_start_point):
        edge_end_point = self.parse_node()
        return Edge(edge_start_point, edge_end_point)

    def find_node_head(self):
        return self.buffer.find_node_head()


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
        'Read ahead of the character of current position from self.text'
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

    def find_meta_span(self):
        meta_span = ""
        c = self.text[self.pos]
        if c != LEFT_PAREN:
            raise SexpParseError(f"Wrong characters after `^` in '{ self.text[:self.pos]}*{self.text[self.pos:]}")
        self.accept(LEFT_PAREN)
        meta_span += self.find_node_head()
        self.accept(RIGHT_PAREN)
        return f"({meta_span})"

    def find_consecutive_span(self):
        """"""
        c = self.text[self.pos]
        if c == DOUBLE_QUOTE:
            self.accept(DOUBLE_QUOTE)
            out_str = ""
            while self.peek() != '"':
                c_string = self.next_char()
                out_str += c_string
                if c_string == "\\":
                    out_str += self.next_char()
            self.next_char()
            return f'"{out_str}"'
        elif c == META:
            self.accept(META)
            meta = self.find_meta_span()
            expr = self.find_consecutive_span()
            return [META+meta, expr]
        else:
            out_inner = ""
            # if c != "\\":
            #     out_inner += c

            # TODO: is there a better loop idiom here?
            if not self.is_eoi():
                next_c = self.peek()
                escaped = c == "\\"
                while (not self.is_eoi()) and (
                        escaped or not _is_beginning_control_char(next_c)
                ):
                    if (not escaped) and next_c == "\\":
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
        while self.skip_then_peek() != RIGHT_PAREN:
            if self.skip_then_peek() == LEFT_PAREN:
                break
            found_span = self.find_consecutive_span()
            if isinstance(found_span, list):
                out_head.extend(found_span)
            else:
                out_head.append(found_span)
        return out_head[0] if len(out_head) == 1 else out_head

    def accept(self, string):
        'Check if the string passed in matches indexed raw text'
        self.skip_whitespace() # self.pos ++ if current pos points to a whitespace.
        end = self.pos + len(string)
        if end <= self.length and string == self.text[self.pos:end]:
            self.pos += len(string)
            return
        else:
            raise SexpParseError(f"failed to accept '{string}' in '{ self.text[:self.pos]}*{self.text[self.pos:]}'")

    def end_of_stream(self):
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
    )

if __name__ == "__main__":
    test_string = r"""
(Yield
  (Event.start
    (FindNumNextEvent
      (Event.subject_? (?~= "staff meeting"))
      1L)))
    """
    parser = SexpParser(test_string)
    node = parser.parse()
    nested_dict = node.to_nested_dict()
    print()
    print(node)