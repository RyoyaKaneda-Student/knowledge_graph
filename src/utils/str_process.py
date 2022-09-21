def YesNo(x) -> str:
    return 'Yes' if x else 'No'


def YESNO(x) -> str:
    return 'YES' if x else 'NO'


def blank_or_not(x) -> str:
    return '' if x else 'not'


def blank_or_Not(x) -> str:
    return '' if x else 'Not'


def blank_or_NOT(x) -> str:
    return '' if x else 'NOT'


def line_up_key_value(defined_mark_='=', separate_mark_='.', **kwargs) -> str:
    ss = []
    for key, value in kwargs.items():
        if value is not None:
            s = '{}{}{}'.format(key, defined_mark_, value)
            ss.append(s)
    return separate_mark_.join(ss)


def info_str(s):
    return s.ljust(25).center(40, '=')
