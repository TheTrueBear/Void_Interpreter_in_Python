
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
TT_FLOAT = 'float'

TT_PLUS    = 'plus'
TT_MINUS   = 'minus'
TT_MUL     = 'multiply'
TT_DIV     = 'divide'
TT_MODULO  = 'modulo'
TT_POWER   = 'power'

TT_LPAREN   = 'lparen'
TT_RPAREN   = 'rparen'
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
    "let", # Make a variable
    "const", # Make a constant
    "fun", # Make a function
    "is", # Is
    "if", # If
    "for", # For loop
    "while", # While loop
    "elseif", # Else if
    "else", # Else
    "in", # If something is in something else, e.g: "if "A" in "Are you a bear?"
    "export", # Make something public for files
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
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()
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
            elif self.cc == "=":
                tokens.append(Token(TT_EQ, pos_start=self.pos))
                self.advance()
            elif self.cc == "(":
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.cc == ")":
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()

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

        while self.ct.type in ops:
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

        return res.failure(InvalidSyntaxError(
            "Expected int, float, identifier, '+', '-', or '(', got %s" % tok,
            tok.pos_start, tok.pos_end
        ))
    def power(self):
        return self.bin_op(self.atom, (TT_POWER, ), self.factor)
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

        node = res.register(self.bin_op(self.term, (TT_PLUS, TT_MINUS)))
        if res.error:
            return res.failure(InvalidSyntaxError(
                "Expected int, float, identifier, 'let', '+', '-', or '(', got %s" % self.ct,
                self.ct.pos_start, self.ct.pos_end
            ))
        return res.success(node)

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
class Number():
    def __init__(self, val):
        self.val = val
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
    
    def copy(self):
        copy = Number(self.val)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.ctx)
        return copy

    def add(self, o):
        if isinstance(o, self.__class__):
            return Number(self.val + o.val).set_context(self.ctx), None
    def sub(self, o):
        if isinstance(o, self.__class__):
            return Number(self.val - o.val).set_context(self.ctx), None
    def mul(self, o):
        if isinstance(o, self.__class__):
            return Number(self.val * o.val).set_context(self.ctx), None
    def div(self, o):
        if isinstance(o, self.__class__):
            if o.val == 0: return None, RTError(
                "Division by 0.", o.pos_start, o.pos_end,
                self.ctx
            )
            return Number(self.val / o.val).set_context(self.ctx), None
    def power(self, o):
        if isinstance(o, self.__class__):
            return Number(self.val ** o.val).set_context(self.ctx), None
    def mod(self, o):
        if isinstance(o, self.__class__):
            if o.val == 0: return None, RTError(
                "Modulo by 0.", o.pos_start, o.pos_end, self.ctx
            )
            return Number(self.val % o.val).set_context(self.ctx), None

    def __repr__(self):
        return str(self.val)

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
    def __init__(self):
        self.symbols = {}
        self.parent = None

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

        if error: return res.failure(error)
        return res.success(number.set_pos(node.pos_start, node.pos_end))

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

###############################
# Run                         #
###############################
GLOBAL_SYMBOLS = SymbolTable()

# Default variables
GLOBAL_SYMBOLS._set("void", Number(0))

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
