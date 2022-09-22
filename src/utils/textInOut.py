from logging import Logger

SQLITE_PREFIX = "sqlite:///./"


def output_txt(output_path, text, *, logger: Logger = None):
    with open(output_path, 'w') as f:
        f.write(text)


def add_txt(output_path, new_text, *, logger: Logger = None):
    with open(output_path, 'a') as f:
        f.write(new_text)


def input_txt(output_path, *, logger: Logger = None):
    text = ""
    with open(output_path, 'r') as f:
        line = f.readline()
        while line:
            text = text + line
            line = f.readline()
    return text
