# ga-hls/diagnostics/arff.py

from __future__ import annotations

import json
import math
from typing import List

from .. import treenode
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
    FUNC,
)
from ..lang.internal_encoder import FormulaLayout, FeatureInfo
from ..lang.ast import IntConst, RealConst, Var, Subscript, FuncCall

def _normalize_token(term: str) -> str:
    """
    Normalize a single token used in ARFF:

    - Strip spaces.
    - Turn Int('t') / Int("t") / Real('t') / Real("t") into plain t.
    - Leave everything else as-is (but space-free).
    """
    term = term.replace(" ", "")
    # Collapse Int('var') / Real('var') -> var
    if (term.startswith("Int(") or term.startswith("Real(")) and term.endswith(")"):
        inner = term[term.find("(") + 1 : -1]
        if (
            (inner.startswith("'") and inner.endswith("'"))
            or (inner.startswith('"') and inner.endswith('"'))
        ):
            inner = inner[1:-1]
        return inner
    return term

def _normalize_row_str(row: str) -> str:
    """
    Normalize a full arrf_str row: split, normalize each token, re-join.
    """
    tokens = row.replace(" ", "").split(",")
    tokens = [_normalize_token(t) for t in tokens]
    return ",".join(tokens)

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def _attach_features_to_layout(attrs: List[str], layout: FormulaLayout | None) -> None:
    """
    Given the list of ARFF attribute declarations (without the '@attribute' prefix)
    and a FormulaLayout, populate layout.features with FeatureInfo objects that
    map attribute names (e.g. NUM2, RELATIONALS1, TERM0, QUANTIFIERS0) to
    AST position indices.

    This uses simple, schema-specific heuristics:
      - QUANTIFIERSk  -> k-th quantifier (role == QUANTIFIER)
      - LOGICALSk     -> k-th logical connective (role == LOGICAL_CONNECTIVE)
      - RELATIONALSk  -> k-th relational operator (role == RELATION_OP)
      - NUMk          -> numeric literal child of k-th relational
      - TERMk         -> term child (Var/Subscript/FuncCall) of k-th relational
    """
    if layout is None:
        return

    layout.features.clear()

    positions = layout.positions
    if not positions:
        return

    # Precompute sorted indices and some role-based lists
    sorted_idxs = sorted(positions.keys())
    quant_idxs = [i for i in sorted_idxs if positions[i].role == "QUANTIFIER"]
    logical_idxs = [i for i in sorted_idxs if positions[i].role == "LOGICAL_CONNECTIVE"]
    relop_idxs = [i for i in sorted_idxs if positions[i].role == "RELATION_OP"]

    # Build per-relop slots: (relop_pos, numeric_child_pos, term_child_pos)
    relop_slots: list[dict[str, int | None]] = []

    for r_idx in relop_idxs:
        relop_pos = positions[r_idx]
        prefix = relop_pos.path + "." if relop_pos.path else ""
        num_pos: int | None = None
        term_pos: int | None = None
        started = False

        for idx in sorted_idxs:
            if idx <= r_idx:
                continue
            p = positions[idx]
            if prefix and p.path.startswith(prefix):
                started = True
                node = p.node
                if num_pos is None and isinstance(node, (IntConst, RealConst)):
                    num_pos = idx
                if term_pos is None and isinstance(node, (Var, Subscript, FuncCall)):
                    term_pos = idx
            else:
                if started:
                    # We reached nodes outside this relop's subtree
                    break

        relop_slots.append({"relop": r_idx, "num": num_pos, "term": term_pos})

    def _kind_from_decl(decl: str) -> str:
        # Very simple: if it says NUMERIC, we treat it as numeric
        return "NUMERIC" if "NUMERIC" in decl.upper() else "NOMINAL"

    for att in attrs:
        # att is something like 'NUM2 NUMERIC' or 'QUANTIFIERS0 {ForAll, Exists}'
        if not att.strip():
            continue
        parts = att.split()
        name = parts[0]
        decl = " ".join(parts[1:]) if len(parts) > 1 else ""
        kind = _kind_from_decl(decl)
        pos_index: int | None = None

        # QUANTIFIERSk, LOGICALSk, RELATIONALSk, NUMk, TERMk
        def _parse_index(prefix: str) -> int | None:
            if not name.startswith(prefix):
                return None
            suffix = name[len(prefix):]
            if not suffix.isdigit():
                return None
            return int(suffix)

        # Quantifiers
        k = _parse_index("QUANTIFIERS")
        if k is not None and k < len(quant_idxs):
            pos_index = quant_idxs[k]

        # Logical connectives
        k = _parse_index("LOGICALS")
        if k is not None and k < len(logical_idxs):
            pos_index = logical_idxs[k]

        # Relational operators
        k = _parse_index("RELATIONALS")
        if k is not None and k < len(relop_slots):
            pos_index = relop_slots[k]["relop"]

        # Numeric thresholds
        k = _parse_index("NUM")
        if k is not None and k < len(relop_slots):
            num_pos = relop_slots[k]["num"]
            if num_pos is not None:
                pos_index = num_pos

        # Term positions
        k = _parse_index("TERM")
        if k is not None and k < len(relop_slots):
            term_pos = relop_slots[k]["term"]
            if term_pos is not None:
                pos_index = term_pos

        # If we couldn't map the attribute, leave position_index = -1
        layout.features.append(
            FeatureInfo(
                arff_name=name,
                kind=kind,
                position_index=-1 if pos_index is None else pos_index,
            )
        )

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
        'NUM': 0,
        'SINGALS': 0,
        'TERM': 0
    }
    raw_terminators = list(set(treenode.get_terminators(seed)))

    # Build a set of operator-like tokens that should NEVER be treated as terms
    operator_tokens = set(
        str(x).replace(" ", "")
        for family in (
            QUANTIFIERS,
            RELATIONALS,
            EQUALS,
            ARITHMETICS,
            MULDIV,
            EXP,
            LOGICALS,
            NEG,
            FUNC,
        )
        for x in family
    )

    terminators: list[str] = []
    for value in raw_terminators:
        # Skip numeric leaves
        if isinstance(value, (int, float)):
            continue
        tok = _normalize_token(str(value))
        # Drop anything that is actually an operator/binder, not a term
        if tok in operator_tokens:
            continue
        terminators.append(tok)

    # Deduplicate while preserving order
    terminators = list(dict.fromkeys(terminators))

    for tok in formulae:
        tnorm = _normalize_token(tok)
        if not tnorm:
            continue
        # Skip operators / function names
        if tnorm in operator_tokens:
            continue
        # Skip pure numbers
        if tnorm.isnumeric() or isfloat(tnorm):
            continue
        # Add any new term-like tokens
        if tnorm not in terminators:
            terminators.append(tnorm)

    ret = []
    for term in formulae:
        term = _normalize_token(term)
        if term in QUANTIFIERS:
            qstring = str(QUANTIFIERS)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            qstring = qstring.replace(' ', '')
            ret.append(f'QUANTIFIERS{count_op["QUANTIFIERS"]} {qstring}')
            count_op['QUANTIFIERS'] = count_op['QUANTIFIERS'] + 1
        elif term in RELATIONALS:
            rel_domain = "{<,>,<=,>=}"
            ret.append(f'RELATIONALS{count_op["RELATIONALS"]} {rel_domain}')
            count_op["RELATIONALS"] += 1
        elif term in EQUALS:
            qstring = str(EQUALS)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            qstring = qstring.replace(' ', '')
            ret.append(f'EQUALS{count_op["EQUALS"]} {qstring}')
            count_op['EQUALS'] = count_op['EQUALS'] + 1
        elif term in ARITHMETICS:
            qstring = str(ARITHMETICS)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            qstring = qstring.replace(' ', '')
            ret.append(f'ARITHMETICS{count_op["ARITHMETICS"]} {qstring}')
            count_op['ARITHMETICS'] = count_op['ARITHMETICS'] + 1
        elif term in MULDIV:
            qstring = str(MULDIV)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            qstring = qstring.replace(' ', '')
            ret.append(f'MULDIV{count_op["MULDIV"]} {qstring}')
            count_op['MULDIV'] = count_op['MULDIV'] + 1
        elif term in EXP:
            qstring = str(EXP)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            qstring = qstring.replace(' ', '')
            ret.append(f'EXP{count_op["EXP"]} {qstring}')
            count_op['EXP'] = count_op['EXP'] + 1
        elif term in LOGICALS:
            qstring = str(LOGICALS)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            qstring = qstring.replace(' ', '')
            ret.append(f'LOGICALS{count_op["LOGICALS"]} {qstring}')
            count_op['LOGICALS'] = count_op['LOGICALS'] + 1
        elif term in NEG:
            qstring = str(NEG)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            qstring = qstring.replace(' ', '')
            ret.append(f'NEG{count_op["NEG"]} {qstring}')
            count_op['NEG'] = count_op['NEG'] + 1
        elif term in terminators:
            qstring = str(terminators)
            qstring = qstring.replace('\'', '')
            qstring = '{'+qstring[1:-1]+'}'
            qstring = qstring.replace(' ', '')
            ret.append(f'TERM{count_op["TERM"]} {qstring}')
            count_op['TERM'] = count_op['TERM'] + 1
        elif term.isnumeric() or isfloat(term):
            ret.append(f'NUM{count_op["NUM"]} NUMERIC')
            qstring = qstring.replace(' ', '')
            count_op['NUM'] = count_op['NUM'] + 1
        elif term in FUNC:
            qstring = str(FUNC)
            qstring = qstring.replace('\'', '')
            qstring = qstring.replace(']', '}')
            qstring = qstring.replace('[', '{')
            qstring = qstring.replace(' ', '')
            ret.append(f'FUNC{count_op["FUNC"]} {qstring}')
            count_op['FUNC'] = count_op['FUNC'] + 1
    return ret

def write_dataset_all( path: str, now, seed, population, seed_ch, unknown, unsats, sats, entire_dataset, layout):
    # entire_dataset = sats + unsats + unknown
    entire_dataset = list()
    [entire_dataset.append(x) for x in unknown if (x not in entire_dataset)]
    [entire_dataset.append(x) for x in unsats if (x not in entire_dataset)]
    [entire_dataset.append(x) for x in sats if (x not in entire_dataset)]
    per_cut = 1.0

    try:
        chstr = population[0].arrf_str()
    except Exception as e:
        print(f"[ga-hls][dataset_all] population[0].arrf_str() failed: {e!r}")
        chstr = str(seed_ch)

    chstr_norm = _normalize_row_str(chstr)
    attrs = build_attributes(seed, chstr_norm.split(","))
    _attach_features_to_layout(attrs, layout)

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
            ch_str_norm = _normalize_row_str(ch_str)
            f.write(ch_str_norm)
            f.write(f",{chromosome.madeit.upper()}\n")
    return filename_str

def write_dataset_qty(path: str, now, seed, seed_ch, sats: List, unsats: List, unknown: List, layout: FormulaLayout, per_cut: float) -> str:
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

    chstr_norm = _normalize_row_str(chstr)
    attrs = build_attributes(seed, chstr_norm.split(","))
    _attach_features_to_layout(attrs, layout)

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
            ch_str_norm = _normalize_row_str(chromosome.arrf_str())
            f.write(ch_str_norm)
            f.write(f",{chromosome.madeit.upper()}\n")
        for chromosome in unsats:
            ch_str_norm = _normalize_row_str(chromosome.arrf_str())
            f.write(ch_str_norm)
            f.write(f",{chromosome.madeit.upper()}\n")
        for chromosome in unknown:
            ch_str_norm = _normalize_row_str(chromosome.arrf_str())
            f.write(ch_str_norm)
            f.write(f",{chromosome.madeit.upper()}\n")
    return filename_str

