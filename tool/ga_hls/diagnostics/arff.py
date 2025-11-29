# tool/ga_hls/diagnostics/arff.py

from __future__ import annotations

import json
import math
from typing import List

from .. import treenode, defs
from ..individual import (
    Individual,
    QUANTIFIERS,
    RELATIONALS,
    EQUALS,
    ARITHMETICS,
    MULDIV,
    EXP,
    LOGICALS,
    NEG,
    IMP,
    FUNC,
)

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def build_attributes(seed, formulae: list):
    count_op = {
        'QUANTIFIERS': 0,
        'RELATIONALS': 0,
        'EQUALS': 0,
        'ARITHMETICS': 0,
        'MULDIV': 0,
        'EXP': 0,
        'LOGICALS': 0,
        'FUNC': 0,
        'NEG': 0,
        'IMP': 0,
        'NUM': 0,
        'SINGALS': 0,
        'TERM': 0
    }
    terminators = list(set(treenode.get_terminators(seed)))
    terminators = [value for value in terminators if not isinstance(value, int) and not isinstance(value, float)]
    ret = []
    for term in formulae:
        if term in QUANTIFIERS:
            qstring = str(QUANTIFIERS)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            ret.append(f'QUANTIFIERS{count_op["QUANTIFIERS"]} {qstring}')
            count_op['QUANTIFIERS'] = count_op['QUANTIFIERS'] + 1
        elif term in RELATIONALS:
            qstring = str(RELATIONALS)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            ret.append(f'RELATIONALS{count_op["RELATIONALS"]} {qstring}')
            count_op['RELATIONALS'] = count_op['RELATIONALS'] + 1
        elif term in EQUALS:
            qstring = str(EQUALS)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            ret.append(f'EQUALS{count_op["EQUALS"]} {qstring}')
            count_op['EQUALS'] = count_op['EQUALS'] + 1
        elif term in ARITHMETICS:
            qstring = str(ARITHMETICS)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            ret.append(f'ARITHMETICS{count_op["ARITHMETICS"]} {qstring}')
            count_op['ARITHMETICS'] = count_op['ARITHMETICS'] + 1
        elif term in MULDIV:
            qstring = str(MULDIV)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            ret.append(f'MULDIV{count_op["MULDIV"]} {qstring}')
            count_op['MULDIV'] = count_op['MULDIV'] + 1
        elif term in EXP:
            qstring = str(EXP)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            ret.append(f'EXP{count_op["EXP"]} {qstring}')
            count_op['EXP'] = count_op['EXP'] + 1
        elif term in LOGICALS:
            qstring = str(LOGICALS)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            ret.append(f'LOGICALS{count_op["LOGICALS"]} {qstring}')
            count_op['LOGICALS'] = count_op['LOGICALS'] + 1
        elif term in NEG:
            qstring = str(NEG)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            ret.append(f'NEG{count_op["NEG"]} {qstring}')
            count_op['NEG'] = count_op['NEG'] + 1
        elif term in IMP:
            qstring = str(IMP)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            ret.append(f'IMP{count_op["IMP"]} {qstring}')
            count_op['IMP'] = count_op['IMP'] + 1
        elif term in terminators:
            qstring = str(terminators)
            qstring = qstring.replace('\'', '')
            qstring = '{'+qstring[1:-1]+'}'
            ret.append(f'TERM{count_op["TERM"]} {qstring}')
            count_op['TERM'] = count_op['TERM'] + 1
        elif term.isnumeric() or isfloat(term):
            ret.append(f'NUM{count_op["NUM"]} NUMERIC')
            count_op['NUM'] = count_op['NUM'] + 1
        elif term in FUNC:
            qstring = str(FUNC)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            ret.append(f'FUNC{count_op["FUNC"]} {qstring}')
            count_op['FUNC'] = count_op['FUNC'] + 1
    return ret

def write_dataset_all(path: str, now, seed, population, seed_ch, unknown, unsats, sats, entire_dataset):
    # entire_dataset = sats + unsats + unknown
    entire_dataset = list()
    [entire_dataset.append(x) for x in unknown if (x not in entire_dataset)]
    [entire_dataset.append(x) for x in unsats if (x not in entire_dataset)]
    [entire_dataset.append(x) for x in sats if (x not in entire_dataset)]
    per_cut = 1.0

    try:
        chstr = population[0].arrf_str()
    except Exception as e:
        chstr = str(seed_ch)

    attrs = build_attributes(seed,chstr.split(','))

    nowstr = f'{now}'.replace(' ', '_')
    nowstr = nowstr.replace(':', '_')
    filename_str = '{}/dataset_qty_{}_all.arff'.format(path, nowstr)
    with open(filename_str, 'w') as f:
        nowstr = f'{nowstr}\n'
        nowstr = nowstr.replace(' ', '_')
        s = f'@relation all.{nowstr}\n'
        f.write(s)
        f.write('\n')
        for att in attrs:
            f.write(f'@attribute {att}\n')

        f.write('@attribute VEREDICT {TRUE, FALSE, UNKNOWN}\n')
        f.write('\n')
        f.write('@data\n')
        for chromosome in entire_dataset:
            ch_str = chromosome.arrf_str()
            f.write(ch_str)
            f.write(f",{chromosome.madeit.upper()}\n")
    return filename_str

def write_dataset_threshold(path: str, unknown, unsats, sats, now):
    unsats.sort(key=lambda x : x.sw_score, reverse=True)
    if len(unsats) > len(sats):
        unsats = unsats[:len(sats)]
    else:
        sats = sats[:len(unsats)]

    with open('{}/dataset_{}.arff'.format(path, now), 'w') as f:
        f.write('@relation all.generationall\n')
        f.write('\n')
        f.write('@attribute QUANTIFIERS {ForAll, Exists}\n')
        f.write('@attribute VARIABLE {s}\n')
        f.write('@attribute RELATIONALS {<,>,<=,>=, =}\n')
        f.write('@attribute NUMBER NUMERIC\n')
        f.write('@attribute LOGICALS1 {And, Or}\n')
        f.write('@attribute VARIABLE1 {s}\n')
        f.write('@attribute RELATIONALS1 {<,>,<=,>=, =}\n')
        f.write('@attribute NUMBER1 NUMERIC\n')
        f.write('@attribute IMP1 {Implies}\n')
        f.write('@attribute SIGNALS {signal_2(s),signal_4(s)}\n')
        f.write('@attribute RELATIONALS2 {<,>,<=,>=, =}\n')
        f.write('@attribute NUMBER2 NUMERIC\n')
        f.write('@attribute LOGICALS2 {And, Or}\n')
        f.write('@attribute SIGNALS1 {signal_2(s),signal_4(s)}\n')
        f.write('@attribute RELATIONALS3 {<,>,<=,>=, =}\n')
        f.write('@attribute NUMBER3 NUMERIC\n')
        f.write('@attribute VEREDICT {TRUE, FALSE, UNKNOWN}\n')
        f.write('\n')
        f.write('@data\n')
        for chromosome in sats:
            ch_str = str(chromosome)
            ch_str = ch_str.replace(' ', ',')
            ch_str = ch_str.replace(',s,In,(', ',')
            ch_str = ch_str.replace('),Implies,(', ',Implies,')
            f.write(ch_str[:-1])
            f.write(f",{chromosome.madeit.upper()}\n")
        for chromosome in unsats:
            ch_str = str(chromosome)
            ch_str = ch_str.replace(' ', ',')
            ch_str = ch_str.replace(',s,In,(', ',')
            ch_str = ch_str.replace('),Implies,(', ',Implies,')
            f.write(ch_str[:-1])
            f.write(f",{chromosome.madeit.upper()}\n")

def write_dataset_qty(path: str, now, seed, seed_ch, sats: List, unsats: List, unknown: List, per_cut: float) -> str:
    sats.sort(key=lambda x : x.sw_score, reverse=True)
    unsats.sort(key=lambda x : x.sw_score, reverse=True)
    unknown.sort(key=lambda x : x.sw_score, reverse=True)

    min_len = min([len(sats), len(unsats), len(unknown)])
    per_qty = math.ceil(min_len * per_cut)
    sats = sats
    unsats = unsats
    unknown = unknown
    if len(sats) > per_qty:
        sats = sats[:per_qty]
    if len(unsats) > per_qty:
        unsats = unsats[:per_qty]
    if len(unknown) > per_qty:
        unknown = unknown[:per_qty]

    try:
        chstr = sats[0].arrf_str()
    except Exception as e:
        chstr = str(seed_ch)

    attrs = build_attributes(seed,chstr.split(','))

    nowstr = f'{now}'.replace(' ', '_')
    nowstr = nowstr.replace(':', '_')
    filename_str = '{}/dataset_qty_{}_per{:0>3d}.arff'.format(path, nowstr, int(per_cut*100))
    with open(filename_str, 'w') as f:
        nowstr = f'{nowstr}\n'
        nowstr = nowstr.replace(' ', '_')
        s = f'@relation all.{nowstr}\n'
        f.write(s)
        f.write('\n')
        for att in attrs:
            f.write(f'@attribute {att}\n')

        f.write('@attribute VEREDICT {TRUE, FALSE, UNKNOWN}\n')
        f.write('\n')
        f.write('@data\n')
        for chromosome in sats:
            # print('writing sats')
            ch_str = chromosome.arrf_str()
            # ch_str = ch_str.replace(' ', ',')
            # ch_str = ch_str.replace(',t,In,(', ',')
            # ch_str = ch_str.replace('),Implies,(', ',Implies,')
            f.write(ch_str)
            f.write(f",{chromosome.madeit.upper()}\n")
        for chromosome in unsats:
            # print('writing unsats')
            ch_str = chromosome.arrf_str()
            # ch_str = ch_str.replace(' ', ',')
            # ch_str = ch_str.replace(',t,In,(', ',')
            # ch_str = ch_str.replace('),Implies,(', ',Implies,')
            f.write(ch_str)
            f.write(f",{chromosome.madeit.upper()}\n")
        for chromosome in unknown:
            # print('writing unsats')
            ch_str = chromosome.arrf_str()
            # ch_str = ch_str.replace(' ', ',')
            # ch_str = ch_str.replace(',t,In,(', ',')
            # ch_str = ch_str.replace('),Implies,(', ',Implies,')
            f.write(ch_str)
            f.write(f",{chromosome.madeit.upper()}\n")
    return filename_str

