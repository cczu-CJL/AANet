#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from hmanet.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "hmanetPlans"

    overwrite_plans = {
        'hmanetTrainerV2_2': ["hmanetPlans", "hmanetPlansisoPatchesInVoxels"], # r
        'hmanetTrainerV2': ["hmanetPlansnonCT", "hmanetPlansCT2", "hmanetPlansallConv3x3",
                            "hmanetPlansfixedisoPatchesInVoxels", "hmanetPlanstargetSpacingForAnisoAxis",
                            "hmanetPlanspoolBasedOnSpacing", "hmanetPlansfixedisoPatchesInmm", "hmanetPlansv2.1"],
        'hmanetTrainerV2_warmup': ["hmanetPlans", "hmanetPlansv2.1", "hmanetPlansv2.1_big", "hmanetPlansv2.1_verybig"],
        'hmanetTrainerV2_cycleAtEnd': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_cycleAtEnd2': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_reduceMomentumDuringTraining': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_graduallyTransitionFromCEToDice': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_independentScalePerAxis': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_Mish': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_Ranger_lr3en4': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_fp32': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_GN': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_momentum098': ["hmanetPlans", "hmanetPlansv2.1"],
        'hmanetTrainerV2_momentum09': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_DP': ["hmanetPlansv2.1_verybig"],
        'hmanetTrainerV2_DDP': ["hmanetPlansv2.1_verybig"],
        'hmanetTrainerV2_FRN': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_resample33': ["hmanetPlansv2.3"],
        'hmanetTrainerV2_O2': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_ResencUNet': ["hmanetPlans_FabiansResUNet_v2.1"],
        'hmanetTrainerV2_DA2': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_allConv3x3': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_ForceBD': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_ForceSD': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_LReLU_slope_2en1': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_lReLU_convReLUIN': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_ReLU': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_ReLU_biasInSegOutput': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_ReLU_convReLUIN': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_lReLU_biasInSegOutput': ["hmanetPlansv2.1"],
        #'hmanetTrainerV2_Loss_MCC': ["hmanetPlansv2.1"],
        #'hmanetTrainerV2_Loss_MCCnoBG': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_Loss_DicewithBG': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_Loss_Dice_LR1en3': ["hmanetPlansv2.1"],
        'hmanetTrainerV2_Loss_Dice': ["hmanetPlans", "hmanetPlansv2.1"],
        'hmanetTrainerV2_Loss_DicewithBG_LR1en3': ["hmanetPlansv2.1"],
        # 'hmanetTrainerV2_fp32': ["hmanetPlansv2.1"],
        # 'hmanetTrainerV2_fp32': ["hmanetPlansv2.1"],
        # 'hmanetTrainerV2_fp32': ["hmanetPlansv2.1"],
        # 'hmanetTrainerV2_fp32': ["hmanetPlansv2.1"],
        # 'hmanetTrainerV2_fp32': ["hmanetPlansv2.1"],

    }

    trainers = ['hmanetTrainer'] + ['hmanetTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'hmanetTrainerNewCandidate24_2',
        'hmanetTrainerNewCandidate24_3',
        'hmanetTrainerNewCandidate26_2',
        'hmanetTrainerNewCandidate27_2',
        'hmanetTrainerNewCandidate23_always3DDA',
        'hmanetTrainerNewCandidate23_corrInit',
        'hmanetTrainerNewCandidate23_noOversampling',
        'hmanetTrainerNewCandidate23_softDS',
        'hmanetTrainerNewCandidate23_softDS2',
        'hmanetTrainerNewCandidate23_softDS3',
        'hmanetTrainerNewCandidate23_softDS4',
        'hmanetTrainerNewCandidate23_2_fp16',
        'hmanetTrainerNewCandidate23_2',
        'hmanetTrainerVer2',
        'hmanetTrainerV2_2',
        'hmanetTrainerV2_3',
        'hmanetTrainerV2_3_CE_GDL',
        'hmanetTrainerV2_3_dcTopk10',
        'hmanetTrainerV2_3_dcTopk20',
        'hmanetTrainerV2_3_fp16',
        'hmanetTrainerV2_3_softDS4',
        'hmanetTrainerV2_3_softDS4_clean',
        'hmanetTrainerV2_3_softDS4_clean_improvedDA',
        'hmanetTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'hmanetTrainerV2_3_softDS4_radam',
        'hmanetTrainerV2_3_softDS4_radam_lowerLR',

        'hmanetTrainerV2_2_schedule',
        'hmanetTrainerV2_2_schedule2',
        'hmanetTrainerV2_2_clean',
        'hmanetTrainerV2_2_clean_improvedDA_newElDef',

        'hmanetTrainerV2_2_fixes', # running
        'hmanetTrainerV2_BN', # running
        'hmanetTrainerV2_noDeepSupervision', # running
        'hmanetTrainerV2_softDeepSupervision', # running
        'hmanetTrainerV2_noDataAugmentation', # running
        'hmanetTrainerV2_Loss_CE', # running
        'hmanetTrainerV2_Loss_CEGDL',
        'hmanetTrainerV2_Loss_Dice',
        'hmanetTrainerV2_Loss_DiceTopK10',
        'hmanetTrainerV2_Loss_TopK10',
        'hmanetTrainerV2_Adam', # running
        'hmanetTrainerV2_Adam_hmanetTrainerlr', # running
        'hmanetTrainerV2_SGD_ReduceOnPlateau', # running
        'hmanetTrainerV2_SGD_lr1en1', # running
        'hmanetTrainerV2_SGD_lr1en3', # running
        'hmanetTrainerV2_fixedNonlin', # running
        'hmanetTrainerV2_GeLU', # running
        'hmanetTrainerV2_3ConvPerStage',
        'hmanetTrainerV2_NoNormalization',
        'hmanetTrainerV2_Adam_ReduceOnPlateau',
        'hmanetTrainerV2_fp16',
        'hmanetTrainerV2', # see overwrite_plans
        'hmanetTrainerV2_noMirroring',
        'hmanetTrainerV2_momentum09',
        'hmanetTrainerV2_momentum095',
        'hmanetTrainerV2_momentum098',
        'hmanetTrainerV2_warmup',
        'hmanetTrainerV2_Loss_Dice_LR1en3',
        'hmanetTrainerV2_NoNormalization_lr1en3',
        'hmanetTrainerV2_Loss_Dice_squared',
        'hmanetTrainerV2_newElDef',
        'hmanetTrainerV2_fp32',
        'hmanetTrainerV2_cycleAtEnd',
        'hmanetTrainerV2_reduceMomentumDuringTraining',
        'hmanetTrainerV2_graduallyTransitionFromCEToDice',
        'hmanetTrainerV2_insaneDA',
        'hmanetTrainerV2_independentScalePerAxis',
        'hmanetTrainerV2_Mish',
        'hmanetTrainerV2_Ranger_lr3en4',
        'hmanetTrainerV2_cycleAtEnd2',
        'hmanetTrainerV2_GN',
        'hmanetTrainerV2_DP',
        'hmanetTrainerV2_FRN',
        'hmanetTrainerV2_resample33',
        'hmanetTrainerV2_O2',
        'hmanetTrainerV2_ResencUNet',
        'hmanetTrainerV2_DA2',
        'hmanetTrainerV2_allConv3x3',
        'hmanetTrainerV2_ForceBD',
        'hmanetTrainerV2_ForceSD',
        'hmanetTrainerV2_ReLU',
        'hmanetTrainerV2_LReLU_slope_2en1',
        'hmanetTrainerV2_lReLU_convReLUIN',
        'hmanetTrainerV2_ReLU_biasInSegOutput',
        'hmanetTrainerV2_ReLU_convReLUIN',
        'hmanetTrainerV2_lReLU_biasInSegOutput',
        'hmanetTrainerV2_Loss_DicewithBG_LR1en3',
        #'hmanetTrainerV2_Loss_MCCnoBG',
        'hmanetTrainerV2_Loss_DicewithBG',
        # 'hmanetTrainerV2_Loss_Dice_LR1en3',
        # 'hmanetTrainerV2_Ranger_lr3en4',
        # 'hmanetTrainerV2_Ranger_lr3en4',
        # 'hmanetTrainerV2_Ranger_lr3en4',
        # 'hmanetTrainerV2_Ranger_lr3en4',
        # 'hmanetTrainerV2_Ranger_lr3en4',
        # 'hmanetTrainerV2_Ranger_lr3en4',
        # 'hmanetTrainerV2_Ranger_lr3en4',
        # 'hmanetTrainerV2_Ranger_lr3en4',
        # 'hmanetTrainerV2_Ranger_lr3en4',
        # 'hmanetTrainerV2_Ranger_lr3en4',
        # 'hmanetTrainerV2_Ranger_lr3en4',
        # 'hmanetTrainerV2_Ranger_lr3en4',
        # 'hmanetTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
