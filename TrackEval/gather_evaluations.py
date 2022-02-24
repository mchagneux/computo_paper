import pandas as pd
import os 
import matplotlib.pyplot as plt
import pickle 
import numpy as np
from collections import defaultdict

eval_dir_part_1 = None
eval_dir_all = None
eval_dir_short = None
long_segments_names = None
indices = None

def get_det_values(index_start=0, index_stop=-1):

    results_for_det = pd.read_csv(os.path.join(eval_dir_short,'surfrider-test','ours_EKF_1_12fps_v0_tau_0','pedestrian_detailed.csv'))
    results_det = results_for_det.loc[:,['DetRe___50','DetPr___50', 'HOTA_TP___50','HOTA_FN___50','HOTA_FP___50']].iloc[index_start:index_stop]
    results_det.columns = ['hota_det_re','hota_det_pr','hota_det_tp','hota_det_fn','hota_det_fp']
    hota_det_re = results_det['hota_det_re']
    hota_det_pr = results_det['hota_det_pr']
    hota_det_tp = results_det['hota_det_tp']
    hota_det_fn = results_det['hota_det_fn']
    hota_det_fp = results_det['hota_det_fp']

    denom_hota_det_re = hota_det_tp + hota_det_fn 
    denom_hota_det_pr = hota_det_tp + hota_det_fp 

    hota_det_re_cb = (hota_det_re * denom_hota_det_re).sum() / denom_hota_det_re.sum()
    hota_det_pr_cb = (hota_det_pr * denom_hota_det_pr).sum() / denom_hota_det_pr.sum()

    return [hota_det_re_cb, hota_det_pr_cb]

def get_table_det():

    table_values = [get_det_values(index_start, index_stop) for (index_start, index_stop) in zip(indices[:-1],indices[1:])]
    table_values.append(get_det_values())
    table = f'\\hline \\\[-1.8ex]\n'
    table += f'$S_1$ & {100*table_values[0][0]:.1f} & {100*table_values[0][1]:.1f} \\\\\n'
    table += f'$S_2$ & {100*table_values[1][0]:.1f} & {100*table_values[1][1]:.1f} \\\\\n'
    table += f'$S_3$ & {100*table_values[2][0]:.1f} & {100*table_values[2][1]:.1f} \\\\\n'
    table += f'$All$ & {100*table_values[3][0]:.1f} & {100*table_values[3][1]:.1f} \\\\\n'

    print(table)

def get_ass_re_values(tracker_name):

    results_p1 =   pd.read_csv(os.path.join(eval_dir_part_1,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))
    all_results =  pd.read_csv(os.path.join(eval_dir_all,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))

    ass_re_p1 = _round(results_p1.loc[2,'AssRe___50'])
    ass_re_p2 = _round(all_results.loc[2,'AssRe___50'])
    ass_re_p3 = _round(all_results.loc[3,'AssRe___50'])
    ass_re_cb = _round(all_results.loc[4,'AssRe___50'])

    return [ass_re_p1,ass_re_p2,ass_re_p3,ass_re_cb]

def plot_errors(tracker_names, tracker_new_names=None, last_index=-1, filename='detailed_errors'):

    all_results = {tracker_name: pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv')) for tracker_name in tracker_names}

    predicted_counts = {k:v.loc[:,['Correct_IDs___50','Missing_IDs___50','Redundant_IDs___50','False_IDs___50']].iloc[:last_index] for k,v in all_results.items()}

    # predicted_counts['GT_IDs'] = all_results[tracker_names[0]].loc[:,['GT_IDs']].iloc[:-1]
    # print(all_results['sort'])
    # count_errors.index = all_results[tracker_names[0]]['seq'][:-1]
    # count_errors.columns = tracker_new_names
    # idxmins = count_errors.abs().idxmin(axis=1)
    


    # # count_errors_relative.drop(labels=[29],inplace=True)

    # # print(count_errors)
    # # fig, ax = plt.subplots(1,1,figsize=(10,10))

    positions = [len(predicted_counts.values())-x-3.5 for x in range(len(predicted_counts.values()))]

    fig, ax = plt.subplots()

    for (position, v) in zip(positions, predicted_counts.values()):
        v.index = all_results[tracker_names[0]]['seq'][:last_index]
        v.columns = ['Correct','Missing','Redundant','False']
        v.plot.bar(stacked=True, position=position, ax=ax, width=0.2, color=['green','silver','orange','red'], edgecolor='black',linewidth=0.1)
    
    gt_ids = all_results[tracker_names[0]].loc[:,['GT_IDs']].iloc[:last_index]
    gt_ids.index = all_results[tracker_names[0]]['seq'].iloc[:last_index]
    gt_ids.columns = ['Ground truth']
    gt_ids.plot.bar(position=len(predicted_counts.values())-2.5,ax=ax,width=0.2,color='black')


    # # plt.vlines(x=[17,24],ymin=-10,ymax=10)
    # # plt.plot(idxmins)

    # plt.hlines(y=[0],xmin=-1,xmax=len(count_errors.index))
    # plt.ylabel('$err_s$')
    # plt.xlabel('$s$')
    # plt.xticks(np.arange(len(count_errors.index)),count_errors.index, rotation='vertical')
    # plt.grid(True,axis='y')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.xlabel('$Sequence$')
    plt.ylabel('$Count$')
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(False)

    plt.tight_layout()
    plt.autoscale(True)
    # plt.show()
    plt.savefig(f'{filename}.pdf',format='pdf')

def get_summary(results, index_start=0, index_stop=-1):

    results = results.loc[:,['Correct_IDs___50','Redundant_IDs___50','False_IDs___50','Missing_IDs___50','Fused_IDs___50', 'GT_IDs','HOTA_TP___50','AssRe___50']].iloc[index_start:index_stop]

    results.columns = ['correct','redundant','false','missing','mingled','gt','hota_tp','ass_re']

    ass_re = results['ass_re']
    hota_tp = results['hota_tp']


    ass_re_cb = (ass_re * hota_tp).sum() / hota_tp.sum()


    redundant = results['redundant']
    false = results['false']
    missing = results['missing']
    # mingled = results['mingled'] 
    gt = results['gt']
    count_error = false + redundant - missing

    summary = dict()
    summary['missing'], summary['missing_mean'], summary['missing_std'] = f'{missing.sum()}',f'{missing.mean():.1f}',f'{missing.std():.1f}'
    summary['false'], summary['false_mean'], summary['false_std'] = f'{false.sum()}', f'{false.mean():.1f}', f'{false.std():.1f}'
    summary['redundant'], summary['redundant_mean'], summary['redundant_std'] = f'{redundant.sum()}', f'{redundant.mean():.1f}', f'{redundant.std():.1f}'
    summary['gt'] = f'{gt.sum()}'
    summary['ass_re_cb'], summary['ass_re_mean'], summary['ass_re_std'] = f'{100*ass_re_cb:.1f}',f'{100*ass_re.mean():.1f}',f'{100*ass_re.std():.1f}'
    summary['count_error'], summary['count_error_mean'], summary['count_error_std'] = f'{count_error.sum()}',f'{count_error.mean():.1f}',f'{count_error.std():.1f}'


    return summary 

def get_summaries(results, sequence_names):

    summaries = dict()

    for (sequence_name, index_start, index_stop) in zip(sequence_names, indices[:-1],indices[1:]):

        summaries[sequence_name] = get_summary(results, index_start, index_stop)
    
    summaries['All'] = get_summary(results)

    return summaries

def get_ass_pre_values(tracker_name):
    results = pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))

    results = results.loc[:,['AssPr___50']].iloc[-1]

    print(f'{100*results.sum():.1f}')

def table_values_for_sequence(summaries, sequence_name):
    summary = summaries[sequence_name]
    table_values = f"${sequence_name}$ " + \
                   f"& " + \
                   f"{summary['ass_re_cb']} ({summary['ass_re_std']}) & " + \
                   f"{summary['missing']} ({summary['missing_std']}) & " + \
                   f"{summary['false']} ({summary['false_std']}) & " + \
                   f"{summary['redundant']} ({summary['redundant_std']}) & \\\ \n\hhline{{~~~~~}} & "

    return table_values

def get_table_values(tracker_name, tracker_new_name):


    results = pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))
    sequence_names = ['S_1','S_2','S_3']

    summaries = get_summaries(results, sequence_names)


    table = f"\\multirow{{ 3 }}{{*}}  {{{tracker_new_name}}} & "

    for sequence_name in sequence_names:
        table += table_values_for_sequence(summaries, sequence_name)
    table += table_values_for_sequence(summaries, 'All')[:-17] + "\hline \\\[-1.8ex]"
    
    print(table)

def get_count_errors(tracker_name):
    results = pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))
    sequence_names = ['S_1','S_2','S_3']

    summary = get_summaries(results, sequence_names)['All']

    print(f"{summary['count_error']}, ({summary['count_error_mean']}, {summary['count_error_std']})")

def read_mot_results_file(filename):
    raw_results =  np.loadtxt(filename, delimiter=',')
    if raw_results.ndim == 1: raw_results = np.expand_dims(raw_results,axis=0)
    tracklets = defaultdict(list) 
    for result in raw_results:
        track_id = int(result[1])
        frame_id = int(result[0])
        left, top, width, height = result[2:6]
        center_x = left + width/2
        center_y = top + height/2 
        tracklets[track_id].append((frame_id, center_x, center_y))

    tracklets = list(tracklets.values())

    return sorted(tracklets, key=lambda x:x[0][0])

def hyperparameters():
    tau_values = [i for i in range(4,10)]
    versions = ['v0','v2_3','v2_5','v2_7']
    fig, (ax0, ax1, ax2) = plt.subplots(3,1)

    for version in versions:
        tracker_names = [f'ours_EKF_1_12fps_{version}_tau_{tau}' for tau in tau_values]
        all_results = {tracker_name: pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv')).iloc[[-1]] for tracker_name in tracker_names}

        n_missing = []
        n_false = []
        n_redundant = []
        for tracker_name, tracker_results in all_results.items():
            missing = (tracker_results['GT_IDs'] - tracker_results['Correct_IDs___50'])[27]
            false = tracker_results['False_IDs___50'][27]
            redundant = tracker_results['Redundant_IDs___50'][27]

            n_missing.append(missing)
            n_false.append(false)
            n_redundant.append(redundant)

        ax0.scatter(tau_values, n_missing, label=version)
        ax0.plot(tau_values, n_missing, label=version, linestyle='dashed')
        # ax0.set_xlabel('$\\tau$')
        ax0.set_ylabel('$N_{missing}$')

        ax1.scatter(tau_values, n_false, label=version)
        ax1.plot(tau_values, n_false, label=version, linestyle='dashed')

        # ax1.set_xlabel('$\\tau$')
        ax1.set_ylabel('$N_{incorrect}$')

        ax2.scatter(tau_values, n_redundant, label=version)
        ax2.plot(tau_values, n_redundant, label=version, linestyle='dashed')
        ax2.set_xlabel('$\\tau$')
        ax2.set_ylabel('$N_{redundant}$')
    # handles, labels = ax2.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center')
    # plt.autoscale(True)

    plt.tight_layout()
    plt.savefig('comparison.pdf',format='pdf')
    
if __name__ == '__main__':

    fps = 12
    fps = f'{fps}fps'


    gt_dir_short = f'TrackEval/data/gt/surfrider_short_segments_{fps}/surfrider-test' 
    eval_dir_short = f'TrackEval/data/trackers/surfrider_short_segments_{fps}' 
    
    long_segments_names = ['part_1_1','part_1_2','part_2','part_3']
    indices = [0,16,19,27]
    indices_det = [0,17,24,38]

    # get_table_values_averages('fairmot','FairMOT')
    # get_table_values_averages('fairmot_cleaned','FairMOT*')
    # # get_table_values_averages('sort','SORT')


    # get_table_values('fairmot','$FairMOT$')
    # get_table_values('fairmot_cleaned','$FairMOT^{*}$')
    # get_table_values('sort','$SORT$')
    # get_table_values('ours_EKF_1_12fps_v2_7_tau_5','$Ours$')

    # hyperparameters()


    # get_count_errors('fairmot_cleaned')
    # get_count_errors('sort')
    # get_count_errors('ours_EKF_1_12fps_v2_7_tau_5')


    # get_ass_pre_values('ours_EKF_1_12fps_v2_7_tau_5')
    # get_ass_pre_values('sort')
    # get_ass_pre_values('fairmot')
    # get_ass_pre_values('fairmot_cleaned')


    # get_table_det()

    # get_table_values_averages('ours')

    # get_table_values_absolute(['fairmot_cleaned','sort','ours_EKF_1_12fps_v0_tau_6'],['$FairMOT^{*}$','SORT','$Ours_{v0}$',])

    # compare_with_humans('comptages_auterrive2021.csv',tracker_names=['fairmot_cleaned','sort',f'ours_{fps}_{tau}'])

    # plot_errors(['ours_EKF_1_12fps_v2_7_tau_5','sort','fairmot_cleaned'], last_index=indices[1])
    # plot_errors(['ours_EKF_1_12fps_v2_5_tau_6','sort','fairmot_cleaned'], filename='last_detailed_errors')

    # tau_values = [i for i in range(1,10)]
    # versions = ['v2_5','v2_7']
    # for version in versions:
    #     tracker_names = ['ours_EKF_1_12fps_v0_tau_0'] + [f'ours_EKF_1_12fps_{version}_tau_{tau}' for tau in tau_values]
    #     # plot_errors(tracker_names,filename=version)
    #     optimal_tau(tracker_names,filename=version)

    # hyperparameters()