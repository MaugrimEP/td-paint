from dataclasses import field


def return_factory(a):
    return field(default_factory=lambda: a)

