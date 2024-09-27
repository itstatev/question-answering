import re


def parse_fn(response):
    return re.findall(r'(\w+) -> (\w+) -> (\w+)', response)
