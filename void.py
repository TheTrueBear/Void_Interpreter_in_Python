
"""
Interpreter for W Sharp programming language.

Don't copy or redistribute this program.
"""

###############################
# Imports                     #
###############################
import string

###############################
# Generate a string w/ arrows #
###############################
def string_with_arrows(text, pos_start, pos_end):
    result = ''
    idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
    idx_end = text.find('\n', idx_start + 1)
    if idx_end < 0: idx_end = len(text)
    line_count = pos_end.ln - pos_start.ln + 1
    for i in range(line_count):
        line = text[idx_start:idx_end]
        col_start = pos_start.col if i == 0 else 0
        col_end = pos_end.col if i == line_count - 1 else len(line) - 1
        result += line + '\n'
        result += ' ' * col_start + '^' * (col_end - col_start)
        idx_start = idx_end
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0: idx_end = len(text)
    return result.replace('\t', '')

###############################
# Constants                   #
###############################
DIGITS         = "0123456789"
ALPH           = string.ascii_letters
ID_ALLOW       = ALPH + DIGITS + "_"
ID_START_ALLOW = ALPH + "_"

TT_INT    = 'integer'
TT_FLOAT  = 'float'
TT_STRING = 'string'
TT_LIST   = 'list'

TT_PLUS    = 'plus'
TT_MINUS   = 'minus'
TT_MUL     = 'multiply'
TT_DIV     = 'divide'
TT_MODULO  = 'modulo'
TT_POWER   = 'power'

TT_LBRACKET = 'lbracket'
TT_RBRACKET = 'rbracket'
TT_LPAREN   = 'lparen'
TT_RPAREN   = 'rparen'
TT_LBRACE   = 'lbrace'
TT_RBRACE   = 'rbrace'

TT_ARROW    = 'arrow'
TT_COMMA    = 'comma'
TT_KEYWORD  = 'keyword'
TT_IDENTIFY = 'identifier'
TT_EQ       = 'equals'
TT_EOF      = 'eof'

TT_EE = 'is equal to'
TT_NE = 'not equal to'
TT_GT = 'greater than'
TT_GE = 'greater then or equal to'
TT_LT = 'less than'
TT_LE = 'less than or equal to'

KEYWORDS = [
    "let",   # Make a variable
    "not",   # Invert conditional statement
    "or",    # 'a or b': return true if either is true
    "and",   # 'a and b': return true if both are true

    "if",    # Used for if statements
    "elif",  # Else if statement
    "else",  # Else statement
    "then",  # Placeholder for if statements

    "for",   # Used for a for loop.
    "to",    # Used for a for loop, to specify end value
    "step",  # Used in a for loop to change the step
    "while", # Used for a while loop

    "fun",   # Define a function
]

###############################
# Position                    #
###############################
class Position():
    def __init__(self, idx, ln, col, fn, ftxt) -> None:
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt
    def advance(self, cc=None):
        self.idx += 1
        self.col += 1
        if cc in ("\n", ";"):
            self.ln += 1
            self.col = 0
    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

###############################
# Token                       #
###############################
class Token():
    def __init__(self, _type, _value=None, pos_start=None, pos_end=None):
        self.type = _type
        self.value = _value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()
        if pos_end:
            self.pos_end = pos_end.copy()

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value: return f'{self.type}: {self.value}'
        return f'{self.type}'

###############################
# Error                       #
###############################
class Error():
    def __init__(self, name, details, pos_start, pos_end):
        self.name = name
        self.details = details
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        result = f'{self.name}: {self.details}\n'
        result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
        result += f'\n\n{string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)}'
        return result

class IllegalCharError(Error):
    def __init__(self, details, pos_start, pos_end):
        super().__init__("Illegal character", details, pos_start, pos_end)
class InvalidSyntaxError(Error):
    def __init__(self, details, pos_start, pos_end):
        super().__init__("Invalid syntax", details, pos_start, pos_end)
class RTError(Error):
    def __init__(self, details, pos_start, pos_end, ctx):
        super().__init__("Runtime error", details, pos_start, pos_end)
        self.ctx = ctx

    def generate_traceback(self):
        result = ""
        pos = self.pos_start
        ctx = self.ctx

        while ctx:
            result = f'    File "{pos.fn}", line {str(pos.ln + 1)}, in "{ctx.name}"\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent
        return "Traceback (most recent call last):\n" + result

    def __repr__(self):
        result = self.generate_traceback()
        result += f'{self.name}: {self.details}'
        result += f'\n\n{string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)}'
        return result
class ExpectedCharError(Error):
    def __init__(self, details, pos_start, pos_end):
        super().__init__("Expected character", details, pos_start, pos_end)

###############################
# Lexer                       #
###############################
class Lexer():
    def __init__(self, text, fn):
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.cc = None
        self.advance()

    def advance(self):
        self.pos.advance(self.cc)
        self.cc = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    ###############################
    # Make a number
    def make_number(self):
        pos_start = self.pos.copy()
        num_str = ''
        dots = 0

        while self.cc != None and self.cc in DIGITS + ".":
            if self.cc == ".":
                if dots >= 1: break
                dots += 1
                num_str += '.'
            else:
                num_str += self.cc
            self.advance()

        if dots == 0:
            return Token(TT_INT, int(num_str), pos_start=pos_start, pos_end=self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start=pos_start, pos_end=self.pos)

    # Make ID or KEY
    def make_id(self):
        id_str = ""
        pos_start = self.pos.copy()

        while self.cc != None and self.cc in ID_ALLOW:
            id_str += self.cc
            self.advance()

        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFY
        return Token(tok_type, id_str, pos_start, self.pos)
    
    # Make not equals
    def make_ne(self):
        pos_start = self.pos.copy()
        self.advance()
        if self.cc == "=":
            self.advance()
            return Token(TT_NE, pos_start, self.pos), None
        return None, ExpectedCharError("Character '=' expected after '!'.", pos_start, self.pos)
    def make_equals(self):
        pos_start = self.pos.copy()
        self.advance()
        tok_type = TT_EQ
        if self.cc == "=":
            self.advance()
            tok_type = TT_EE
        return Token(tok_type, None, pos_start, self.pos)
    def make_lt(self):
        pos_start = self.pos.copy()
        self.advance()
        tok_type = TT_LT
        if self.cc == "=":
            self.advance()
            tok_type = TT_LE
        return Token(tok_type, None, pos_start, self.pos)
    def make_gt(self):
        pos_start = self.pos.copy()
        self.advance()
        tok_type = TT_GT
        if self.cc == "=":
            self.advance()
            tok_type = TT_GE
        return Token(tok_type, None, pos_start, self.pos)
    def make_sub_or_arrow(self):
        pos_start = self.pos.copy()
        self.advance()
        tok_type = TT_MINUS
        if self.cc == ">":
            self.advance()
            tok_type = TT_ARROW
        return Token(tok_type, None, pos_start, self.pos)
    
    # Make a string
    def make_string(self):
        string = ""
        pos_start = self.pos.copy()
        escape_char = False
        self.advance()

        escape_chars = {
            "n": "\n",
            "t": "\t",
        }

        while self.cc != None and (self.cc != "\"" or escape_char):
            if escape_char:
                escape_char = False
                string += escape_chars.get(self.cc, self.cc)
            else:
                if self.cc == "\\":
                    escape_char = True
                else:
                    string += self.cc
            self.advance()
        
        self.advance()
        return Token(TT_STRING, string, pos_start, self.pos)

    # Tokenize
    def tokenize(self):
        tokens = []

        while self.cc != None:
            if self.cc in " \t":
                self.advance()

            # Numbers
            elif self.cc in DIGITS:
                tokens.append(self.make_number())

            # Letters
            elif self.cc in ID_START_ALLOW:
                tokens.append(self.make_id())

            # Operators
            elif self.cc == "+":
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.cc == "-":
                tokens.append(self.make_sub_or_arrow())
            elif self.cc == "*":
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.cc == "/":
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.cc == "^":
                tokens.append(Token(TT_POWER, pos_start=self.pos))
                self.advance()
            elif self.cc == "%":
                tokens.append(Token(TT_MODULO, pos_start=self.pos))
                self.advance()
            elif self.cc == "(":
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.cc == ")":
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            elif self.cc == ",":
                tokens.append(Token(TT_COMMA, pos_start=self.pos))
                self.advance()
            elif self.cc == "\"":
                tokens.append(self.make_string())
            
            # Conditionals
            elif self.cc == "=":
                tokens.append(self.make_equals())
            elif self.cc == "!":
                tok, error = self.make_ne()
                if error: return [], error
                tokens.append(tok)
            elif self.cc == "<":
                tokens.append(self.make_lt())
            elif self.cc == ">":
                tokens.append(self.make_gt())

            # Errors
            else:
                pos_start = self.pos.copy()
                char = self.cc
                self.advance()
                return [], IllegalCharError("'%s'"%char, pos_start, self.pos)

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

###############################
# Nodes                       #
###############################
class NumberNode():
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    def __repr__(self):
        return f'{self.tok}'
class BinOpNode():
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
        self.pos_start = self.left.pos_start
        self.pos_end = self.right.pos_end
    def __repr__(self):
        return f'({self.left}, {self.op}, {self.right})'
class UnaryOpNode():
    def __init__(self, op, node):
        self.op = op
        self.node = node
        self.pos_start = self.op.pos_start
        self.pos_end = self.node.pos_end
    def __repr__(self):
        return f'({self.op}, {self.node})'

class StringNode():
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    def __repr__(self):
        return f'{self.tok}'
        
class VarAccessNode():
    def __init__(self, name):
        self.name = name
        self.pos_start = name.pos_start
        self.pos_end = name.pos_end
class VarAssignNode():
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.pos_start = self.name.pos_start
        self.pos_end = self.value.pos_end

class IfNode():
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.else_case or self.cases[-1][0]).pos_end

class ForNode():
    def __init__(self, name, start, end, step, body):
        self.name = name
        self.start = start
        self.end = end
        self.step = step
        self.body = body
        
        self.pos_start = self.name.pos_start
        self.pos_end = self.body.pos_end
class WhileNode():
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

        self.pos_start = self.condition.pos_start
        self.pos_end = self.body.pos_end

class FuncDefNode():
    def __init__(self, name, args, body):
        self.name = name
        self.args = args
        self.body = body

        if self.name:
            self.pos_start = self.name.pos_start
        elif len(self.args) > 0:
            self.pos_start = self.args[0].pos_start
        else:
            self.pos_start = self.body.pos_start
        self.pos_end = self.body.pos_end
class CallNode():
    def __init__(self, to_call, args):
        self.to_call = to_call
        self.args = args

        self.pos_start = self.to_call.pos_start
        self.pos_end = self.to_call.pos_end
        if len(self.args) > 0:
            self.pos_end = self.args[-1].pos_end

###############################
# Parse result                #
###############################
class ParseResult():
    def __init__(self):
        self.error = None
        self.node = None
        self.advancements = 0

    def register_advancement(self):
        self.advancements += 1
    def register(self, res):
        self.advancements += res.advancements
        if res.error: self.error = res.error
        return res.node
    def success(self, node):
        self.node = node
        return self
    def failure(self, error):
        if not self.error or self.advancements == 0:
            self.error = error
        return self

###############################
# Parser                      #
###############################
class Parser():
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.ct = None
        self.advance()
    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens): self.ct = self.tokens[self.tok_idx]
        return self.ct
    def bin_op(self, func, ops, func2=None):
        res = ParseResult()
        func2 = func if func2 == None else func2
        left = res.register(func())
        if res.error: return res

        while self.ct.type in ops or (self.ct.type, self.ct.value) in ops:
            op = self.ct
            res.register_advancement()
            self.advance()
            right = res.register(func2())
            if res.error: return res
            left = BinOpNode(left, op, right)

        return res.success(left)

    ###############################
    # Grammar rules               #
    ###############################
    def atom(self):
        res = ParseResult()
        tok = self.ct

        if tok.type in (TT_INT, TT_FLOAT):
            res.register_advancement()
            self.advance()
            return res.success(NumberNode(tok))

        elif tok.type == TT_IDENTIFY:
            res.register_advancement()
            self.advance()
            return res.success(VarAccessNode(tok))
        elif tok.type == TT_STRING:
            res.register_advancement()
            self.advance()
            return res.success(StringNode(tok))

        elif tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.ct.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    "Expected right parenthesis (')'), got %s" % repr(self.ct),
                    self.ct.pos_start, self.ct.pos_end
                ))
        elif tok.matches(TT_KEYWORD, "if"):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)
        
        elif tok.matches(TT_KEYWORD, "for"):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)
        elif tok.matches(TT_KEYWORD, "while"):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)
        elif tok.matches(TT_KEYWORD, "fun"):
            func_def = res.register(self.func_def())
            if res.error: return res
            return res.success(func_def)

        return res.failure(InvalidSyntaxError(
            "Expected int, float, identifier, 'if', 'for', 'while', 'fun', '+', '-', or '(', got %s" % tok,
            tok.pos_start, tok.pos_end
        ))
    def call(self):
        res = ParseResult()
        atom = res.register(self.atom())
        if res.error: return res

        if self.ct.type == TT_LPAREN:
            # Advance
            res.register_advancement()
            self.advance()
            arg_nodes = []

            if self.ct.type == TT_RPAREN:
                # Advance
                res.register_advancement()
                self.advance()
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error: return res.failure(
                        InvalidSyntaxError("Expected ')', 'var', 'if', 'for', 'while', 'fun', int or float, got %s" % self.ct,
                        self.ct.pos_start, self.ct.pos_end
                    ))
                while self.ct.type == TT_COMMA:
                    # Advance
                    res.register_advancement()
                    self.advance()

                    arg_nodes.append(res.register(self.expr()))
                    if res.error: return res
                
                if self.ct.type != TT_RPAREN:
                    return res.failure(InvalidSyntaxError("Expected ',' or ')', got %s" % self.ct, self.ct.pos_start, self.ct.pos_end))
                
                # Advance
                res.register_advancement()
                self.advance()
            return res.success(CallNode(atom, arg_nodes))
        return res.success(atom)
    def power(self):
        return self.bin_op(self.call, (TT_POWER, ), self.factor)
    def factor(self):
        res = ParseResult()
        tok = self.ct

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        return self.power()
    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV, TT_MODULO))
    def arith_expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))
    def comp_expr(self):
        res = ParseResult()
        if self.ct.matches(TT_KEYWORD, "not"):
            op = self.ct
            res.register_advancement()
            self.advance()
            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(op, node))
        node = res.register(self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_GE, TT_LE)))
        if res.error: return res.failure(InvalidSyntaxError(
            "Expected int, float, identifier, 'not', '+', '-', or '(', got %s" % self.ct,
            self.ct.pos_start, self.ct.pos_end
        ))
        return res.success(node)
    def expr(self):
        res = ParseResult()
        if self.ct.matches(TT_KEYWORD, 'let'):
            res.register_advancement()
            self.advance()
            if self.ct.type != TT_IDENTIFY:
                return res.failure(InvalidSyntaxError(
                    "Expected identifier, got %s" % self.ct,
                    self.ct.pos_start, self.ct.pos_end
                ))

            var_name = self.ct
            res.register_advancement()
            self.advance()
            if self.ct.type != TT_EQ:
                return res.failure(InvalidSyntaxError(
                    "Expected '=', got %s" % self.ct,
                    self.ct.pos_start, self.ct.pos_end
                ))

            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))

        node = res.register(self.bin_op(self.comp_expr, ((TT_KEYWORD, 'and'), (TT_KEYWORD, 'or'))))
        if res.error:
            return res.failure(InvalidSyntaxError(
                "Expected int, float, identifier, 'let', 'fun', 'while', 'for', 'if', '+', '-', or '(', got %s" % self.ct,
                self.ct.pos_start, self.ct.pos_end
            ))
        return res.success(node)
    
    # Read if statement
    def if_expr(self):
        # Set up variables
        res = ParseResult()
        cases = []
        else_case = None

        # Make sure the first keyword is IF
        if not self.ct.matches(TT_KEYWORD, "if"):
            return res.failure(InvalidSyntaxError(
                "Expected 'if', got %s" % self.ct,
                self.ct.pos_start, self.ct.pos_end
            ))

        # Advance
        res.register_advancement()
        self.advance()

        # Get the condition
        condition = res.register(self.expr())
        if res.error: return res

        # TODO: Change this to use curly braces ('{}').
        if not self.ct.matches(TT_KEYWORD, "then"):
            return res.failure(InvalidSyntaxError(
                "Expected 'then', got %s" % self.ct,
                self.ct.pos_start, self.ct.pos_end
            ))
        
        # Advance
        res.register_advancement()
        self.advance()

        # Get the code
        expr = res.register(self.expr())
        if res.error: return res
        cases.append((condition, expr))

        # Read for elif statements
        while self.ct.matches(TT_KEYWORD, "elif"):
            # Advance
            res.register_advancement()
            self.advance()

            # Get the condition
            condition = res.register(self.expr())
            if res.error: return res
            
            # TODO: Change this to use curly braces ('{}').
            if not self.ct.matches(TT_KEYWORD, "then"):
                return res.failure(InvalidSyntaxError(
                    "Expected 'then', got %s" % self.ct,
                    self.ct.pos_start, self.ct.pos_end
                ))
            
            # Advance
            res.register_advancement()
            self.advance()

            # Get the code
            expr = res.register(self.expr())
            if res.error: return res
            cases.append((condition, expr))
        
        # Read for an if statement
        if self.ct.matches(TT_KEYWORD, "else"):
            res.register_advancement()
            self.advance()

            # Get the expression
            else_case = res.register(self.expr())
            if res.error: return res
        
        # Return an IF node
        return res.success(IfNode(cases, else_case))
    # Read for statement
    def for_expr(self):
        res = ParseResult()

        # Make sure it starts with a FOR key
        if not self.ct.matches(TT_KEYWORD, "for"):
            return res.failure(InvalidSyntaxError(
                "Expected 'for', got %s" % self.ct,
                self.ct.pos_start, self.ct.pos_end
            ))

        # Advance
        res.register_advancement()
        self.advance()

        # Make sure you have an identifier
        if self.ct.type != TT_IDENTIFY:
            return res.failure(InvalidSyntaxError("Expected identifier, got %s" % self.ct, self.ct.pos_start, self.ct.pos_end))

        # Get the var name
        var_name = self.ct

        # Advance
        res.register_advancement()
        self.advance()

        # Get the equal sign
        if self.ct.type != TT_EQ:
            return res.failure(InvalidSyntaxError("Expected equal sign, got %s" % self.ct, self.ct.pos_start, self.ct.pos_end))

        # Advance
        res.register_advancement()
        self.advance()

        # Get start value
        start_value = res.register(self.expr())
        if res.error: return res

        # Get the to keyword
        if not self.ct.matches(TT_KEYWORD, "to"):
            return res.failure(InvalidSyntaxError("Expected 'to', got %s" % self.ct, self.ct.pos_start, self.ct.pos_end))
        
        # Advance
        res.register_advancement()
        self.advance()

        # Get the end value
        end_value = res.register(self.expr())
        if res.error: return res

        # Get the step. NOTE: OPTIONAL
        if self.ct.matches(TT_KEYWORD, "step"):
            res.register_advancement()
            self.advance()

            # Get the value
            step_value = res.register(self.expr())
            if res.error: return res
        else:
            step_value = None
        
        # If there is no THEN. TODO: Change this to a curly brace
        if not self.ct.matches(TT_KEYWORD, "then"):
            return res.failure(InvalidSyntaxError("Expected 'then', got %s" % self.ct, self.ct.pos_start, self.ct.pos_end))

        # Advance
        res.register_advancement()
        self.advance()

        # Get the body
        body = res.register(self.expr())
        if res.error: return res

        # Return the node
        return res.success(ForNode(var_name, start_value, end_value, step_value, body))
    # While loop
    def while_expr(self):
        res = ParseResult()

        # Get the while keyword
        if not self.ct.matches(TT_KEYWORD, "while"):
            return res.failure(InvalidSyntaxError("Expected 'while', got %s" % self.ct, self.ct.pos_start, self.ct.pos_end))

        # Advance
        res.register_advancement()
        self.advance()

        # Get the condition
        condition = res.register(self.expr())
        if res.error: return res

        # Get the then keyword. TODO: Change this to a curly brace.
        if not self.ct.matches(TT_KEYWORD, "then"):
            return res.failure(InvalidSyntaxError("Expected 'then', got %s" % self.ct, self.ct.pos_start, self.ct.pos_end))
        
        # Advance
        res.register_advancement()
        self.advance()

        # Get the body
        body = res.register(self.expr())
        if res.error: return res

        # Return the node
        return res.success(WhileNode(condition, body))
    # Function definition
    def func_def(self):
        res = ParseResult()

        # Get the function keyword
        if not self.ct.matches(TT_KEYWORD, "fun"):
            return res.failure(InvalidSyntaxError("Expected 'fun', got %s" % self.ct, self.ct.pos_start, self.ct.pos_end))
        
        # Advance
        res.register_advancement()
        self.advance()

        # Get the name. Optional.
        if self.ct.type == TT_IDENTIFY:
            var_name_tok = self.ct

            # Advance
            res.register_advancement()
            self.advance()

            # Get the left parenthesis
            if self.ct.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError("Expected '(', got %s" % var_name_tok, self.ct.pos_start, self.ct.pos_end))
        # If there isn't a name
        else:
            var_name_tok = None

            # Get the left parenthesis
            if self.ct.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError("Expected identifier or '(', got %s" % var_name_tok, self.ct.pos_start, self.ct.pos_end))
                
        # Advance
        res.register_advancement()
        self.advance()

        # Get arguments. Optional.
        arg_name_toks = []
        if self.ct.type == TT_IDENTIFY:
            arg_name_toks.append(self.ct)

            # Advance
            res.register_advancement()
            self.advance()

            # While the current character is a comma
            while self.ct.type == TT_COMMA:
                # Advance
                res.register_advancement()
                self.advance()

                # Error handling
                if self.ct.type != TT_IDENTIFY:
                    return res.failure(InvalidSyntaxError("Expected identifier, got %s" % self.ct, self.ct.pos_start, self.ct.pos_end))
                
                # Add the identifier to the list and continue
                arg_name_toks.append(self.ct)
                res.register_advancement()
                self.advance()
            
            # Look for a right parenthesis
            if self.ct.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError("Expected ')' or ',', got %s" % self.ct, self.ct.pos_start, self.ct.pos_end))

        # If you don't have a name (anonymous function)
        else:
            # Get the right parenthesis
            if self.ct.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError("Expected identifier or ')', got %s" % self.ct, self.ct.pos_start, self.ct.pos_end))

        # Advance
        res.register_advancement()
        self.advance()

        # TODO: Change this to use curly braces. NOTE: This uses the ARROW operator, which wil show return type eventually.
        if self.ct.type != TT_ARROW:
            return res.failure(InvalidSyntaxError("Expected '->', got %s" % self.ct, self.ct.pos_start, self.ct.pos_end))

        # Advance
        res.register_advancement()
        self.advance()

        # Get the expression
        node_to_return = res.register(self.expr())
        if res.error: return res

        # Return a successful node
        return res.success(FuncDefNode(var_name_tok, arg_name_toks, node_to_return))

    ###############################
    # Parse                       #
    ###############################
    def parse(self):
        res = self.expr()
        if not res.error and self.ct.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                "Expected operator, got %s" % repr(self.ct),
                self.ct.pos_start, self.ct.pos_end
            ))
        return res

###############################
# Runtime result class        #
###############################
class RTResult():
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error: self.error = res.error
        return res.value
    def success(self, value):
        self.value = value
        return self
    def failure(self, error):
        self.error = error
        return self

###############################
# Values                      #
###############################
class Value():
    def __init__(self):
        self.pos_start = None
        self.pos_end = None
        self.ctx = None
    
    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    def set_context(self, ctx=None):
        self.ctx = ctx
        return self
    
    def add(self, o):
        return None, self.illegal_operation(o)
    def sub(self, o):
        return None, self.illegal_operation(o)
    def mul(self, o):
        return None, self.illegal_operation(o)
    def div(self, o):
        return None, self.illegal_operation(o)
    def power(self, o):
        return None, self.illegal_operation(o)
    def mod(self, o):
        return None, self.illegal_operation(o)
    
    # Comparison
    def comparison_ee(self, o):
        return None, self.illegal_operation(o)
    def comparison_ne(self, o):
        return None, self.illegal_operation(o)
    def comparison_gt(self, o):
        return None, self.illegal_operation(o)
    def comparison_lt(self, o):
        return None, self.illegal_operation(o)
    def comparison_ge(self, o):
        return None, self.illegal_operation(o)
    def comparison_le(self, o):
        return None, self.illegal_operation(o)
    def comparison_and(self, o):
        return None, self.illegal_operation(o)
    def comparison_or(self, o):
        return None, self.illegal_operation(o)
    def notted(self):
        return None, self.illegal_operation()
    def is_true(self):
        return False

    def copy(self):
        raise Exception("No copy method defined")
    def illegal_operation(self, o=None):
        if not o: o = self
        return RTError("Illegal operation!", self.pos_start, self.pos_end, self.ctx)
    def execute(self, args):
        return RTError(f'Can\'t execute type {self.__class__.__name__}', self.pos_start, self.pos_end, self.ctx)
class Number(Value):
    def __init__(self, val):
        self.val = val
        self.pos_start = None
        self.pos_end = None
        self.ctx = None
    def copy(self):
        copy = Number(self.val)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.ctx)
        return copy

    # Arithmetic
    def add(self, o):
        if isinstance(o, self.__class__):
            return Number(self.val + o.val).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    def sub(self, o):
        if isinstance(o, self.__class__):
            return Number(self.val - o.val).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    def mul(self, o):
        if isinstance(o, self.__class__):
            return Number(self.val * o.val).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    def div(self, o):
        if isinstance(o, self.__class__):
            if o.val == 0: return None, RTError(
                "Division by 0.", o.pos_start, o.pos_end,
                self.ctx
            )
            return Number(self.val / o.val).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    def power(self, o):
        if isinstance(o, self.__class__):
            return Number(self.val ** o.val).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    def mod(self, o):
        if isinstance(o, self.__class__):
            if o.val == 0: return None, RTError(
                "Modulo by 0.", o.pos_start, o.pos_end, self.ctx
            )
            return Number(self.val % o.val).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    
    # Comparison
    def comparison_ee(self, o):
        if isinstance(o, self.__class__):
            return Number(int(self.val == o.val)).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    def comparison_ne(self, o):
        if isinstance(o, self.__class__):
            return Number(int(self.val != o.val)).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    def comparison_gt(self, o):
        if isinstance(o, self.__class__):
            return Number(int(self.val > o.val)).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    def comparison_lt(self, o):
        if isinstance(o, self.__class__):
            return Number(int(self.val < o.val)).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    def comparison_ge(self, o):
        if isinstance(o, self.__class__):
            return Number(int(self.val >= o.val)).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    def comparison_le(self, o):
        if isinstance(o, self.__class__):
            return Number(int(self.val <= o.val)).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    def comparison_and(self, o):
        if isinstance(o, self.__class__):
            return Number(int(self.val and o.val)).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    def comparison_or(self, o):
        if isinstance(o, self.__class__):
            return Number(int(self.val or o.val)).set_context(self.ctx), None
        else:
            return None, Value.illegal_operation(self.pos_start, self.pos_end)
    def notted(self):
        return Number((self.val - 1) % 2).set_context(self.ctx), None

    def is_true(self):
        return self.val != 0

    def __repr__(self):
        return str(self.val)
class Function(Value):
    def __init__(self, name, body_node, arg_names):
        super().__init__()
        self.name = name or "<anonymous function>"
        self.body_node = body_node
        self.arg_names = arg_names
    
    def execute(self, args):
        res = RTResult()
        interpreter = Interpreter()

        new_context = Context(self.name, self.ctx, self.pos_start)
        new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)

        if len(args) != len(self.arg_names):
            return res.failure(RTError(
                f'An incorrect number of arguments were provided to function {self.name}. ({abs(len(args)-len(self.arg_names))} too {"few" if len(args) < len(self.arg_names) else "many"})',
                self.pos_start, self.pos_end, self.ctx
            ))
        
        for i in range(len(args)):
            arg_name = self.arg_names[i]
            arg_value = args[i]
            arg_value.set_context(new_context)
            new_context.symbol_table._set(arg_name, arg_value)

        value = res.register(interpreter.visit(self.body_node, new_context))
        if res.error: return res
        return res.success(value)
    
    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names)
        copy.set_context(self.ctx)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy
    def __repr__(self): return f'<function {self.name}>'
class String(Value):
    def __init__(self, value):
        super().__init__()
        self.val = value
    def add(self, o):
        if isinstance(o, String):
            return String(self.val + o.val).set_context(self.ctx), None
        return None, Value.illegal_operation(self, o)
    def is_true(self):
        return len(self.val) != 0
    def copy(self):
        copy = String(self.val)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.ctx)
        return copy
    def __repr__(self): return f'"{self.val}"'

###############################
# Context                     #
###############################
class Context():
    def __init__(self, name, parent=None, parent_entry_pos=None):
        self.name = name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None

###############################
# Symbol table                #
###############################
class SymbolTable():
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def _get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent._get(name)
        return value
    def _set(self, name, value):
        self.symbols[name] = value
    def _del(self, name):
        del self.symbols[name]

###############################
# Interpreter                 #
###############################
class Interpreter():
    def visit(self, node, ctx):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, ctx)
    def no_visit_method(self, node, ctx):
        raise Exception(f'No visit method for {type(node).__name__}. Contact a developer for this to be fixed.')

    ###############################
    # Visits
    def visit_NumberNode(self, node, ctx):
        return RTResult().success(
            Number(node.tok.value).set_context(ctx).set_pos(node.pos_start, node.pos_end)
        )
    def visit_BinOpNode(self, node, ctx):
        rtres = RTResult()
        left = rtres.register(self.visit(node.left, ctx))
        if rtres.error: return rtres
        right = rtres.register(self.visit(node.right, ctx))
        if rtres.error: return rtres


        res, error = None, None
        # Arithmetic operators
        if node.op.type == TT_PLUS:
            res, error = left.add(right)
        elif node.op.type == TT_MINUS:
            res, error = left.sub(right)
        elif node.op.type == TT_MUL:
            res, error = left.mul(right)
        elif node.op.type == TT_DIV:
            res, error = left.div(right)
        elif node.op.type == TT_POWER:
            res, error = left.power(right)
        elif node.op.type == TT_MODULO: 
            res, error = left.mod(right)

        # Comparison operators
        elif node.op.type == TT_EE:
            res, error = left.comparison_ee(right)
        elif node.op.type == TT_NE:
            res, error = left.comparison_ne(right)
        elif node.op.type == TT_LT:
            res, error = left.comparison_lt(right)
        elif node.op.type == TT_GT:
            res, error = left.comparison_gt(right)
        elif node.op.type == TT_LE:
            res, error = left.comparison_le(right)
        elif node.op.type == TT_GE:
            res, error = left.comparison_ge(right)
        elif node.op.matches(TT_KEYWORD, "and"):
            res, error = left.comparison_and(right)
        elif node.op.matches(TT_KEYWORD, "or"):
            res, error = left.comparison_or(right)

        if error:
            return rtres.failure(error)
        else:
            return rtres.success(res.set_pos(node.pos_start, node.pos_end))
    def visit_UnaryOpNode(self, node, ctx):
        res = RTResult()
        number = res.register(self.visit(node.node, ctx))
        if res.error: return res

        error = None
        if node.op.type == TT_MINUS:
            number, error = number.mul(Number(-1))
        elif node.op.matches(TT_KEYWORD, "not"):
            number, error = number.notted()

        if error: return res.failure(error)
        return res.success(number.set_pos(node.pos_start, node.pos_end))

    def visit_StringNode(self, node, ctx):
        return RTResult().success(String(node.tok.value).set_context(ctx))

    def visit_VarAccessNode(self, node, ctx):
        res = RTResult()
        var_name = node.name.value
        value = ctx.symbol_table._get(var_name)

        if not value:
            return res.failure(RTError(
                f'Variable {var_name} is not defined.',
                node.pos_start, node.pos_end, ctx
            ))
        
        value = value.copy().set_pos(node.pos_start, node.pos_end)
        return res.success(value)
    def visit_VarAssignNode(self, node, ctx):
        res = RTResult()
        var_name = node.name.value
        value = res.register(self.visit(node.value, ctx))
        if res.error: return res

        ctx.symbol_table._set(var_name, value)
        return res.success(value)
    
    def visit_IfNode(self, node, ctx):
        res = RTResult()

        for condition, expr in node.cases:
            condition_value = res.register(self.visit(condition, ctx))
            if res.error: return res

            if condition_value.is_true():
                expr_value = res.register(self.visit(expr, ctx))
                if res.error: return res
                return res.success(expr_value)
        
        if node.else_case:
            else_value = res.register(self.visit(node.else_case, ctx))
            if res.error: return res
            return res.success(else_value)
        
        return res.success(None)

    def visit_ForNode(self, node, ctx):
        res = RTResult()

        # Get the start value
        start_value = res.register(self.visit(node.start, ctx))
        if res.error: return res

        # Get the end value
        end_value = res.register(self.visit(node.end, ctx))
        if res.error: return res

        # Get the step value
        if node.step:
            step_value = res.register(self.visit(node.step, ctx))
            if res.error: return res
        else:
            step_value = Number(1)

        # Set i
        i = start_value.val
        
        # Run the loop if step is bigger than 1
        if step_value.val >= 0: 
            condition = lambda: i < end_value.val
        else:
            condition = lambda: i > end_value.val

        # Loop while the condition is true
        while condition():
            ctx.symbol_table._set(node.name.value, Number(i))
            i += step_value.val
            res.register(self.visit(node.body, ctx))
            if res.error: return res
        
        # Return success
        return res.success(None)
    def visit_WhileNode(self, node, ctx):
        res = RTResult()

        # Loop through the whole thing
        while True:
            # Condition
            condition = res.register(self.visit(node.condition, ctx))
            if res.error: return res

            # Break once the condition is false
            if not condition.is_true():
                break

            # Run the body
            res.register(self.visit(node.body, ctx))
            if res.error: return res
        
        # Return success
        return res.success(None)

    def visit_FuncDefNode(self, node, ctx):
        res = RTResult()

        func_name = node.name.value if node.name else None
        body_node = node.body
        arg_names = [arg_name.value for arg_name in node.args]
        func_value = Function(func_name, body_node, arg_names).set_context(ctx).set_pos(node.pos_start, node.pos_end)

        if node.name:
            ctx.symbol_table._set(func_name, func_value)
        return res.success(func_value)
    def visit_CallNode(self, node, ctx):
        res = RTResult()
        args = []

        value_to_call = res.register(self.visit(node.to_call, ctx))
        if res.error: return res
        value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)

        for arg_node in node.args:
            args.append(res.register(self.visit(arg_node, ctx)))
            if res.error: return res
        
        return_value = res.register(value_to_call.execute(args))
        if res.error: return res
        return res.success(return_value)

###############################
# Run                         #
###############################
GLOBAL_SYMBOLS = SymbolTable()

# Default variables
GLOBAL_SYMBOLS._set("void", Number(0))
GLOBAL_SYMBOLS._set("true", Number(1))
GLOBAL_SYMBOLS._set("false", Number(0))

# Run
def run(text, fn) -> None:
    # Generate tokens
    lexer = Lexer(text, fn)
    tokens, error = lexer.tokenize()
    if error: return None, error

    # Generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    # Interpert
    interpreter = Interpreter()
    context = Context("<program>")
    context.symbol_table = GLOBAL_SYMBOLS
    res = interpreter.visit(ast.node, context)

    # Return
    return res.value, res.error

"""
Based off of CodePulse's series.
"""