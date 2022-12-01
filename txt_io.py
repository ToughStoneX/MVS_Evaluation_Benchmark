import numpy as np 


def write_evaluation_results(filepath, scan_list, result_dict):
    with open(filepath, 'w') as f:
        f.write(f'{len(scan_list)}\n')
        f_score_list = []
        precision_list = []
        recall_list = []
        for scan in scan_list:
            f_score = result_dict[scan]["f_score"]
            if isinstance(f_score, list):
                f_score = f_score[0]
            f_score_list.append(f_score)

            precision = result_dict[scan]["precision"]
            if isinstance(precision, list):
                precision = precision[0]
            precision_list.append(precision)

            recall = result_dict[scan]["recall"]
            if isinstance(recall, list):
                recall = recall[0]
            recall_list.append(recall)

            f.write(f'{scan} {f_score:.4f} {precision:.4f} {recall:.4f}\n')

        f_score_mean = np.mean(f_score_list)
        precision_mean = np.mean(precision_list)
        recall_mean = np.mean(recall_list)
        f.write(f'mean {f_score_mean:.4f} {precision_mean:.4f} {recall_mean:.4f}\n')

