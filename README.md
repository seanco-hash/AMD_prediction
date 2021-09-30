# AMD Prediction
A 3D vision transformers based on Timesformer from Facebook approach to predict age-related macular (AMD) 
disease features.  
The project is explained in details in the AMD_prediction.pdf

# Data
The data exists on the school servers in the address:  
/cs/labs/dina/seanco/hadassah/  
The folder OCT_output consists the data after conversion on e2e files to .png.
the folder OCT_output/pre_process consists images after croping and centering.
The original e2e files from Hadassah exists on external drive.

# Model:
All the files I wrote are here in git.
The complete project is in school computers:  
/cs/labs/dina/seanco/hadassah/dl_project/AMD

# Pre-trained models:
Available in Timesformer github.

# Trained Models
One can find my trained models for AMD prediction in checkpoints directory in school server:  
/cs/labs/dina/seanco/hadassah/dl_project/AMD/models/checkpoint  
Each checkpoint can be load through the config files.

# Installation:
To use the code one need to follow the installation instructions in Timesformer github from facebook:  
https://github.com/facebookresearch/TimeSformer  
Few packages are used in addition:
requirements.txt file is supplied

# Training
One can run training session by calling train.py with the desired config file. for example  
python3 <proj_path>/train.py --cfg <proj_path>/configs/VIT_8x224_simple_run.yaml 