import argparse


def trans_string_to_boolean(v):
    if v.lower() in ('yes', 'true', 'y', '1'):
        return True
    if v.lower() in ('no', 'n', 'false', '0'):
        return False
    return argparse.ArgumentParser("Boolean value is expected")

def get_entry(tag_sequence, character_sequence):
    NAME = get_name_entry(tag_sequence, character_sequence)
    PROD = get_prod_entry(tag_sequence, character_sequence)
    return NAME, PROD

def get_name_entry(tag_sequence, character_sequence):
    """

    :param tag_sequence:
    :param character_sequence:
    :return:
    """
    length = len(character_sequence)
    NAME = []
    for i, (char, tag) in enumerate(zip(character_sequence, tag_sequence)):
        if tag == 'B_NAME':
            if 'name' in locals().keys():
                NAME.append('name')
                del name
            name = char
            if i + 1 == length:
                NAME.append(name)
        if tag == 'I_NAME':
            name += char
            if i + 1 == length:
                NAME.append(name)
        if tag not in ['I_NAME', 'B_NAME']:
            if 'name' in locals().keys():
                NAME.append(name)
                del name
        continue
    return NAME


def get_prod_entry(tag_sequence, character_sequence):
    length = len(character_sequence)
    PROD = []
    for i, (char, tag) in enumerate(zip(character_sequence, tag_sequence)):
        if tag == 'B_PROD':
            if 'prod' in locals().keys():
                PROD.append('prod')
                del prod
            prod = char
            if i + 1 == length:
                PROD.append(prod)
        if tag == 'I_PROD':
            prod += char
            if i + 1 == length:
                PROD.append(prod)
        if tag not in['I_PROD', 'B_PROD']:
            if 'prod' in locals().keys():
                PROD.append(prod)
                del prod
            continue
    return PROD
