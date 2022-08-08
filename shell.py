# Sys
import sys
import void

# Constants
_ARGV: list = sys.argv

# Shell loop
while True:
    # This means shell script
    text = ""
    file = False
    if len(_ARGV) == 1:
        text = input("ws> ")
    else:
        with open(_ARGV[1]) as f:
            text = f.read()
        file = True

    # Code
    r, e = void.run(text, "<stdin>" if not file else _ARGV[1].split(".")[0])
    if e: print(repr(e))
    else: print(r)

    # Else
    if file: break
