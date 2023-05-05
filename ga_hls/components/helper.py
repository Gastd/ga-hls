import json
import math

from config.ga_params import *

def open_file(path, param='r'):
    try:
        f = open(path, param)
    except FileNotFoundError as e:
        raise FileNotFoundError("Could not open/read file: ", path)
        sys.exit()
    finally:
        return f

def get_file_w_traces(file):
    s = e = -1
    lines = []

    f = open_file(file)
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

    #with open(z3_filepath,'w') as z3check_file:
    z3check_file = open_file(z3_filepath, "w")
    with z3check_file:
        for l in before:
            z3check_file.write(l)
        form_line = (f'\tz3solver.add({nline})\n')
        z3check_file.write('\n')
        # print(form_line)
        z3check_file.write(form_line)
        z3check_file.write('\n')
        for l in after:
            z3check_file.write(l)

def get_line(file):
    # print(f'Running on {os.getcwd()} folder')
    newf_str = ''
    print(f'Running on {file} folder')

    
    #with open(file) as f:
    f = open_file(file)
    with f:
        f.seek(0, 0)
        for l in f:
            if l.find('z3solver.check') > 0:
                break
            else:
                newf_str += l
        # print(f'py file seek at = {len(newf_str)}')
        d1 = newf_str.rfind('\t')
        d2 = newf_str[1:d1-1].rfind('\t')
        # print(f'py file seek at = {d2}')
        first = newf_str
        return first, d2, newf_str.rfind('\n')

def save_file(first, s, e, nline, file):
    dst = 'ga_hls/temp.py'
    # print(f'Running on {self.file_trace} folder')

    #with open(file) as firstfile, open(dst,'w') as secondfile:
    firstfile = open_file(file)
    secondfile = open_file(dst, "w")

    with firstfile, secondfile:
        firstfile.seek(e)
        secondfile.write(first[:s])
        secondfile.write('\n\n')
        secondfile.write(f'\tz3solver.add({nline})\n')
        for l in firstfile:
            secondfile.write(l)

def write_population(generation, path, highest_sat, population, execution_report):

    #with open('{}/{:0>2}.txt'.format(self.path, generation), 'w') as f:
    f = open_file('{}/{:0>2}.txt'.format(path, generation),"w")

    with f:
        f.write('Formula\tFitness\tSatisfied\n')
        if highest_sat:
            f.write('HC: ')
            f.write(str(highest_sat))
            f.write(f'\t{highest_sat.fitness}')
            f.write(f'\t{highest_sat.madeit}')
            f.write('\n')
        for i, chromosome in enumerate(population):
            f.write('{:0>2}'.format(i)+': ')
            f.write(str(chromosome))
            f.write(f'\t{chromosome.fitness}')
            f.write(f'\t{chromosome.madeit}')
            f.write('\n')
    json_object = json.dumps(execution_report, indent=4)

    #with open(f"{self.path}/report.json", "w") as outfile:
    outfile = open_file(f"{path}/report.json","w")
    with outfile:
        outfile.write(json_object)

def save_z3check(s, e, nline):
    file = 'ga_hls/z3check.py'
    print(f'Running on {file} folder')
    form_line = ''
    newf_str2 = ''
    newf_str3 = ''
    
    z3check_file = open_file(file, "r+")
    with z3check_file:
        for l in z3check_file:
            if l.find('z3solver.check') > 0:
                # form_line = l
                print('found: '+l)
                break
            else:
                newf_str2 += l
        d1 = newf_str2.rfind('\n')
        d2 = newf_str2[1:d1-1].rfind('\n')

    z3check_file = open_file(file, "r+")    
    with z3check_file:
        for l in z3check_file:
            if l.find('z3solver.check') > 0:
                # form_line = l
                print('found: '+l)
                break
            else:
                newf_str3 += l
        newf_str2 = newf_str2[1:d2]
        z3check_file.seek(0, 0)
        print(newf_str2)
        z3check_file.write('\n')
        z3check_file.write(newf_str2)
        form_line = (f'\tz3solver.add({nline})\n')
        print(form_line)
        z3check_file.write(form_line)
        z3check_file.write('\n')
        # for l in z3check_file:
        #     z3check_file.write(l)
        # z3check_file.seek(0, 0)
        # firstfile.seek(0, 0)

def store_dataset_qty(per_cut: float, sats, unsats, build_attributes, path, now):
    sats.sort(key=lambda x : x.sw_score, reverse=True)
    unsats.sort(key=lambda x : x.sw_score, reverse=True)

    per_qty = math.ceil(len(sats) * per_cut)
    sats = sats
    unsats = unsats
    if len(sats) > per_qty:
        sats = sats[:per_qty]
    if len(unsats) > per_qty:
        unsats = unsats[:per_qty]

    chstr = str(sats[0])
    chstr = chstr.replace(' ', ',')
    chstr = chstr.replace(',s,In,(', ',')
    chstr = chstr.replace('),Implies,(', ',Implies,')
    chstr = chstr[:-1]
    # print(chstr.split(','))
    attrs = build_attributes(chstr.split(','))
    # for att in attrs:
    #     print(att)

    #with open('{}/dataset_qty_{}_per{}.arff'.format(self.path, self.now, int(per_cut*100)), 'w') as f:
    f = open_file('{}/dataset_qty_{}_per{}.arff'.format(path, now, int(per_cut*100)),"w")
    with f:
        f.write('@relation all.generationall\n')
        f.write('\n')
        for att in attrs:
            f.write(f'@attribute {att}\n')

        # f.write('@attribute QUANTIFIERS {ForAll, Exists}\n')
        # f.write('@attribute VARIABLE {s}\n')
        # f.write('@attribute RELATIONALS {<,>,<=,>=, =}\n')
        # f.write('@attribute NUMBER NUMERIC\n')
        # f.write('@attribute LOGICALS1 {And, Or}\n')
        # f.write('@attribute VARIABLE1 {s}\n')
        # f.write('@attribute RELATIONALS1 {<,>,<=,>=, =}\n')
        # f.write('@attribute NUMBER1 NUMERIC\n')
        # f.write('@attribute IMP1 {Implies}\n')
        # f.write('@attribute SIGNALS {signal_2(s),signal_4(s)}\n')
        # f.write('@attribute RELATIONALS2 {<,>,<=,>=, =}\n')
        # f.write('@attribute NUMBER2 NUMERIC\n')
        # f.write('@attribute LOGICALS2 {And, Or}\n')
        # f.write('@attribute SIGNALS1 {signal_2(s),signal_4(s)}\n')
        # f.write('@attribute RELATIONALS3 {<,>,<=,>=, =}\n')
        # f.write('@attribute NUMBER3 NUMERIC\n')
        f.write('@attribute VEREDICT {TRUE, FALSE}\n')
        f.write('\n')
        f.write('@data\n')
        for chromosome in sats:
            ch_str = str(chromosome)
            ch_str = ch_str.replace(' ', ',')
            ch_str = ch_str.replace(',s,In,(', ',')
            ch_str = ch_str.replace('),Implies,(', ',Implies,')
            f.write(ch_str[:-1])
            f.write(f",{'TRUE' if chromosome.madeit else 'FALSE'}\n")
        for chromosome in unsats:
            ch_str = str(chromosome)
            ch_str = ch_str.replace(' ', ',')
            ch_str = ch_str.replace(',s,In,(', ',')
            ch_str = ch_str.replace('),Implies,(', ',Implies,')
            f.write(ch_str[:-1])
            f.write(f",{'TRUE' if chromosome.madeit else 'FALSE'}\n")

def store_dataset_threshold(unsats,sats,path,now):
    unsats.sort(key=lambda x : x.sw_score, reverse=True)
    if len(unsats) > len(sats):
        unsats = unsats[:len(sats)]
    else:
        sats = sats[:len(unsats)]

    #with open('{}/dataset_{}.arff'.format(self.path, self.now), 'w') as f:
    f = open_file('{}/dataset_{}.arff'.format(path, now), "w")
    with f:
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
        f.write('@attribute VEREDICT {TRUE, FALSE}\n')
        f.write('\n')
        f.write('@data\n')
        for chromosome in sats:
            ch_str = str(chromosome)
            ch_str = ch_str.replace(' ', ',')
            ch_str = ch_str.replace(',s,In,(', ',')
            ch_str = ch_str.replace('),Implies,(', ',Implies,')
            f.write(ch_str[:-1])
            f.write(f",{'TRUE' if chromosome.madeit else 'FALSE'}\n")
        for chromosome in unsats:
            ch_str = str(chromosome)
            ch_str = ch_str.replace(' ', ',')
            ch_str = ch_str.replace(',s,In,(', ',')
            ch_str = ch_str.replace('),Implies,(', ',Implies,')
            f.write(ch_str[:-1])
            f.write(f",{'TRUE' if chromosome.madeit else 'FALSE'}\n")

def dist2VF(generation,population,path,seed_ch,execution_report):
    res = []
    [res.append(x) for x in population if x not in res]
    # print(res)
    
    sats = []
    unsats = []
    for chromosome in res:
        if chromosome.madeit == False:
            unsats.append(chromosome)
        else:
            sats.append(chromosome)

    # print(sats)
    # print(unsats)

    sats.sort(key=lambda x : x.sw_score, reverse=True)
    unsats.sort(key=lambda x : x.sw_score, reverse=True)

    per20 = int(POPULATION_SIZE * .20)
    if len(sats) > per20:
        sats = sats[:per20]
    if len(unsats) > per20:
        unsats = unsats[:per20]

    #with open('{}/{:0>2}_pareto.txt'.format(self.path, generation), 'w') as f:
    f = open_file('{}/{:0>2}_pareto.txt'.format(path, generation), "w")
    with f:    
        f.write('Formula\tSW_Score\tSatisfied\n')
        f.write('SEED: ')
        f.write(str(seed_ch))
        f.write(f'\t-')
        f.write(f'\t-')
        f.write('\n')
        for i, chromosome in enumerate(sats):
            f.write('{:0>2}'.format(i)+': ')
            f.write(str(chromosome))
            f.write(f'\t{chromosome.sw_score}')
            f.write(f'\t{chromosome.madeit}')
            f.write('\n')
        f.write('\n')
        f.write('\n')
        for i, chromosome in enumerate(unsats):
            f.write('{:0>2}'.format(i)+': ')
            f.write(str(chromosome))
            f.write(f'\t{chromosome.sw_score}')
            f.write(f'\t{chromosome.madeit}')
            f.write('\n')
    json_object = json.dumps(execution_report, indent=4)

    #with open(f"{self.path}/report.json", "w") as outfile:
    outfile = open_file(f"{path}/report.json","w")
    with outfile:
        outfile.write(json_object)

def write_hypothesis(path, hypots):
    #with open('{}/hypot.txt'.format(self.path), 'a') as f:
    f = open_file('{}/hypot.txt'.format(path),"a")
    with f:
        for hypot in hypots:
            f.write(f'\t{hypot[1]}\n')