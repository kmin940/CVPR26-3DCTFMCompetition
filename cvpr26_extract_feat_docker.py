"""
Feature extraction evaluation for docker containers

The code was adapted from the MICCAI FLARE Challenge CVPR24_time_eval.py
https://github.com/JunMa11/FLARE

The testing images will be evaluated one by one.

Folder structure:
cvpr26_extract_feat_docker.py
- team_docker
    - teamname.tar.gz # submitted docker containers from participants
- test_demo
    - imgs
        - case1.npz  # testing image
        - case2.npz
        - ...
- demo_features  # feature extraction results
    - case1_features.npz  # features saved here
    - case2_features.npz
    - ...
"""

import os
join = os.path.join
import shutil
import time
import torch
import argparse
from collections import OrderedDict
import pandas as pd

parser = argparse.ArgumentParser('Feature extraction efficiency evaluation for docker containers', add_help=False)
parser.add_argument('-i', '--test_img_path', default='./test_demo/imgs', type=str, help='testing data path')
parser.add_argument('-l', '--labels_root', default='./test_demo/labels', type=str, help='testing data path')
parser.add_argument('-m', '--mask_root', default=None, type=str, help='path to mask directory (for ROI-based diseases only)')
parser.add_argument('-o','--save_path', default='./test_demo/feats', type=str, help='feature extraction output path')
parser.add_argument('-d','--docker_folder_path', default='./team_dockers', type=str, help='team docker path')
parser.add_argument('--target', default='splenomegaly', type=str, help='target disease for feature extraction')
parser.add_argument('--keep_temp', action='store_true', help='keep temporary input/output folders')
args = parser.parse_args()

test_img_path = args.test_img_path
target = args.target
save_path = args.save_path
docker_path = args.docker_folder_path

input_temp = './inputs/'
output_temp = './outputs'
os.makedirs(save_path, exist_ok=True)

dockers = sorted(os.listdir(docker_path))
dockers = [docker for docker in dockers if docker.endswith('.tar.gz')]
df_path = os.path.join(args.labels_root, f'{target}.csv')
df = pd.read_csv(df_path)
test_cases = df['case_id'].tolist()
test_cases.sort()
#test_cases = test_cases[:1] # for debug

available_splits = df['split'].unique().tolist()


for docker in dockers:
    try:
        # # create temp folders for inference one-by-one
        # if os.path.exists(input_temp):
        #     shutil.rmtree(input_temp)
        # if os.path.exists(output_temp):
        #     shutil.rmtree(output_temp)
        os.makedirs(input_temp, exist_ok=True)
        os.makedirs(output_temp, exist_ok=True)

        # copy label directory to input_temp
        shutil.copytree(args.labels_root, os.path.join(input_temp, "labels"), dirs_exist_ok=True)

        # load docker and create a new folder to save feature extraction results
        teamname = docker.split('.')[0].lower()
        print('teamname docker: ', docker)
        os.system('docker image load -i {}'.format(join(docker_path, docker)))
        team_outpath = join(save_path, teamname, target, 'feats_lin_probe')
        # make subdir in save_path for each split
        for split in available_splits:
            os.makedirs(join(team_outpath, split), exist_ok=True)
        if args.mask_root is not None:
            os.makedirs(join(input_temp, "masks"), exist_ok=True)
        # if os.path.exists(team_outpath):
        #     shutil.rmtree(team_outpath)
        os.makedirs(team_outpath, exist_ok=True)
        os.system('chmod -R 777 ./* ')

        metric = OrderedDict()
        metric['CaseName'] = []
        metric['RunningTime'] = []

        # To obtain the running time for each case, testing cases are inferred one-by-one
        for case in test_cases:
            # get corresponding split from df
            split = df.loc[df['case_id'] == case, 'split'].values[0]
            try:
                dst = shutil.copy(join(test_img_path, case), input_temp)
                os.chmod(dst, 0o644)
            except:
                raise Exception(f"Error copying {case} from {join(test_img_path, case)} to {input_temp}. Please check if the file exists and permissions are set correctly.")
            if args.mask_root is not None:
                mask_src = join(args.mask_root, target, case)
                #if os.path.exists(mask_src):
                dst_mask = shutil.copy(mask_src, join(input_temp, "masks"))
                os.chmod(dst_mask, 0o644)
                # else:
                #     print(f"Warning: Mask file {mask_src} not found for {case}. Proceeding without mask.")
            # print(f'============================ copied {case} to {input_temp} from {join(test_img_path, case)}')
            # assert case exists in input_temp
            # assert os.path.exists(join(input_temp, case)), f"Error: {case} not found in {input_temp} after copying. Please check the file and permissions."
            # Docker run command for feature extraction using extract_feat.sh
            # Set MASKS_DIR environment variable if mask_root is provided
            env_var = '-e MASKS_DIR=/workspace/inputs/masks' if args.mask_root is not None else ''
            cmd = 'docker container run --gpus all -m 8G --name {} --rm {} -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh extract_feat.sh" '.format(teamname, env_var, teamname)
            # print(teamname, ' docker command:', cmd, '\n', 'testing image name:', case)

            start_time = time.time()
            os.system(cmd)
            real_running_time = time.time() - start_time
            # print(f"{case} finished! Feature extraction time: {real_running_time}")

            # save metrics
            metric['CaseName'].append(case)
            metric['RunningTime'].append(real_running_time)

            # Remove input file after processing
            os.remove(join(input_temp, case))

            # Copy all outputs from temp folder to team output folder
            # DO NOT remove outputs from docker (keep them in output_temp)
            if os.path.exists(output_temp) and os.listdir(output_temp):
                for output_file in os.listdir(output_temp):
                    src = join(output_temp, output_file)
                    dst = join(team_outpath, split, output_file)
                    try:
                        if os.path.isfile(src):
                            shutil.copy2(src, dst)
                            # print(f"Copied {output_file} to {team_outpath}")
                        elif os.path.isdir(src):
                            if os.path.exists(dst):
                                shutil.rmtree(dst)
                            shutil.copytree(src, dst)
                            # print(f"Copied directory {output_file} to {team_outpath}")
                    except Exception as e:
                        print(f"Error copying {output_file}: {e}")
            else:
                print(f"Warning: No outputs found in {output_temp} for {case}")

            # Save metrics CSV
            metric_df = pd.DataFrame(metric)
            metric_df.to_csv(join(team_outpath, teamname + '_feature_extraction_time.csv'), index=False)

        # Cleanup
        torch.cuda.empty_cache()
        os.system("docker rmi {}:latest".format(teamname))

        # Clean up temp folders unless --keep_temp is specified
        if not args.keep_temp:
            if os.path.exists(input_temp):
                shutil.rmtree(input_temp)
            if os.path.exists(output_temp):
                shutil.rmtree(output_temp)
        else:
            print(f"Keeping temporary folders: {input_temp}, {output_temp}")

    except Exception as e:
        print(f"Error processing {docker}: {e}")
        import traceback
        traceback.print_exc()
