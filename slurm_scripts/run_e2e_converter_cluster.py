import subprocess


WORKDIR = "/cs/labs/dina/seanco/hadassah/dl_project/AMD/datasets"

patients_dir = "/cs/labs/dina/seanco/hadassah/OCT"
output_dir = "/cs/labs/dina/seanco/hadassah/OCT_output"

ask_time = "1-0"
mem = "64000M"
cpus = "16"
gpus = "1"
gpu_mem = "32g"
TRAIN_INTRO = "#!/bin/bash \n" \
        "#SBATCH --mem=" + mem +"\n" \
        "#SBATCH -c" + cpus + "\n" \
        "#SBATCH --gres=gpu:" + gpus + ",vmem:" + gpu_mem + "\n" \
        "#SBATCH --time=" + ask_time + "\n" + \
        "cd /cs/labs/dina/seanco/hadassah/dl_project/AMD\n"
INTRO = "#!/bin/bash \n#SBATCH --mem=" + mem + "\n#SBATCH -c" + cpus + " \n#SBATCH --time=" + ask_time + \
        "\n" + \
        "cd /cs/labs/dina/seanco/hadassah/dl_project/AMD/datasets\n"
PYTHON = "python3 "
E2E_CONVERTER = "convert_e2e_to_jpeg.py "
SEAN_E2E_CONV = "sean_convert_e2e.py "

PRE_PROCESSING = "preprocess_new.py "
TRAIN = 'train.py '

PARAMS = "-i /cs/labs/dina/seanco/hadassah/OCT -o /cs/labs/dina/seanco/hadassah/OCT_output --process_num 1"
S_IDX = " --s_idx "
E_IDX = " --e_idx "

CFG_FILE1 = "--cfg /cs/labs/dina/seanco/hadassah/dl_project/AMD/configs/VIT_8x224_cluster_run.yaml"
CFG_FILE2 = "--cfg /cs/labs/dina/seanco/hadassah/dl_project/AMD/configs/VIT_8x224_cluster_run_num2.yaml"
CFG_FILE3 = "--cfg /cs/labs/dina/seanco/hadassah/dl_project/AMD/configs/VIT_8x224_cluster_run_num3.yaml"
CFG_FILE4 = "--cfg /cs/labs/dina/seanco/hadassah/dl_project/AMD/configs/VIT_8x224_cluster_run_num4.yaml"
CFG_CROPS = "--cfg /cs/labs/dina/seanco/hadassah/dl_project/AMD/configs/VIT_8x224_cluster_run_crops.yaml"
CFG_HR = "--cfg /cs/labs/dina/seanco/hadassah/dl_project/AMD/configs/VIT-HR_16x448_Linear.yaml"
CFG_L = "--cfg /cs/labs/dina/seanco/hadassah/dl_project/AMD/configs/TIMESFORMER_L_cluster_run.yaml"
CFG_FILE = CFG_HR


def train_on_cluster(i=''):
    with open(WORKDIR + '/train_on_cluster' + 'i' + '.sh', 'w') as cur_script:
        cur_script.write(TRAIN_INTRO)
        cur_script.write(PYTHON + TRAIN + CFG_FILE)
        cur_script.write("\n")
        cur_script.close()
    subprocess.run("sbatch " + WORKDIR + '/train_on_cluster' + 'i' + '.sh',
                   shell=True)


def run_on_cluster():
    with open(WORKDIR + '/run_converter_on_cluster.sh', 'w') as cur_script:
        cur_script.write(INTRO)
        cur_script.write(PYTHON + E2E_CONVERTER + PARAMS)
        cur_script.write("\n")
        cur_script.close()
    subprocess.run("sbatch " + WORKDIR + '/run_converter_on_cluster.sh',
                   shell=True)


def run_sean_converter(s, e, i):
    with open(WORKDIR + '/run_converter_on_cluster' + str(i) + '.sh', 'w') as cur_script:
        cur_script.write(INTRO)
        cur_script.write(PYTHON + SEAN_E2E_CONV + PARAMS + S_IDX + str(s) + E_IDX + str(e))
        cur_script.write("\n")
        cur_script.close()
    subprocess.run("sbatch " + WORKDIR + '/run_converter_on_cluster' + str(i) + '.sh',
                   shell=True)


def run_pre_processing():
    with open(WORKDIR + '/run_preprocess_on_cluster.sh', 'w') as cur_script:
        cur_script.write(INTRO)
        cur_script.write(PYTHON + PRE_PROCESSING)
        cur_script.write("\n")
        cur_script.close()
    subprocess.run("sbatch " + WORKDIR + '/run_preprocess_on_cluster.sh', shell=True)


def run_multiple_scripts():
    s_idx = 0
    ep = 100
    i = 0
    while s_idx < 601:
        e_idx = s_idx + ep
        run_sean_converter(s_idx, e_idx, i)
        s_idx += ep
        i += 1


def main():
    #run_multiple_scripts()
    # run_pre_processing()
    train_on_cluster('0')


if __name__ == "__main__":
    main()
