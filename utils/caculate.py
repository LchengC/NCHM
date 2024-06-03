
# calculate p, r, f1
def calculate(all_label_t, all_predt_t, all_clabel_t, epoch, printlog):
    exact_t = [0 for j in range(len(all_label_t))]
    for k in range(len(all_label_t)):
        if all_label_t[k] == 1 and all_label_t[k] == all_predt_t[k]:
            exact_t[k] = 1

    tpi = 0 # Event pairs intra sentence correct
    li = 0  # The number of causal event pairs intra sentence
    pi = 0  # The number of causal event pairs predicted intra sentence
    tpc = 0  # Event pairs inter sentence correct
    lc = 0  # The number of causal event pairs inter sentence
    pc = 0  # The number of causal event pairs predicted inter sentence

    for i in range(len(exact_t)):
        if exact_t[i] == 1:
            if all_clabel_t[i] == 0:
                tpi += 1
            else:
                tpc += 1

        if all_label_t[i] == 1:
            if all_clabel_t[i] == 0:
                li += 1
            else:
                lc += 1

        if all_predt_t[i] == 1:
            if all_clabel_t[i] == 0:
                pi += 1
            else:
                pc += 1

    printlog('\tINTRA-SENTENCE:')
    recli = tpi / li
    preci = tpi / (pi + 1e-9)
    f1cri = 2 * preci * recli / (preci + recli + 1e-9)

    intra = {
        'epoch': epoch,
        'p': preci,
        'r': recli,
        'f1': f1cri
    }
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(tpi, pi, li))
    printlog("\t\tprecision score: {}".format(intra['p']))
    printlog("\t\trecall score: {}".format(intra['r']))
    printlog("\t\tf1 score: {}".format(intra['f1']))

    # CROSS SENTENCE
    reclc = tpc / lc
    precc = tpc / (pc + 1e-9)
    f1crc = 2 * precc * reclc / (precc + reclc + 1e-9)
    cross = {
        'epoch': epoch,
        'p': precc,
        'r': reclc,
        'f1': f1crc
    }

    printlog('\tCROSS-SENTENCE:')
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(tpc, pc, lc))
    printlog("\t\tprecision score: {}".format(cross['p']))
    printlog("\t\trecall score: {}".format(cross['r']))
    printlog("\t\tf1 score: {}".format(cross['f1']))
    return tpi + tpc, pi + pc, li + lc, intra, cross