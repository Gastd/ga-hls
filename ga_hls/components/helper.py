def get_file_w_traces(file):
    s = e = -1
    lines = []
    
    try:
        f = open(file)
    except FileNotFoundError as e:
        raise FileNotFoundError("Could not open/read file: ", file)
        sys.exit()

    with f:
        for l in f:
            lines.append(l)
    # print('lines')
    for idx, l in enumerate(lines):
        if l.find('z3solver.add') >= 0:
            # print(idx, l)
            s = idx
            break
    for idx, l in enumerate(lines):
        # print(l, l.find('z3solver.check'))
        if l.find('z3solver.check') >= 0:
            # print(idx, l)
            e = idx
            break
    # print('lines')
    return s, e, lines

def save_check_wo_traces(start, end, lines, nline, z3_filepath = 'ga_hls/z3check.py'):
    before = lines[:start]
    after = lines[end:]
    with open(z3_filepath,'w') as z3check_file:
        for l in before:
            z3check_file.write(l)
        form_line = (f'\tz3solver.add({nline})\n')
        z3check_file.write('\n')
        # print(form_line)
        z3check_file.write(form_line)
        z3check_file.write('\n')
        for l in after:
            z3check_file.write(l)
