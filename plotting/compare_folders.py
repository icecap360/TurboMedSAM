import glob
import os
complete_dir = '/data/qasim/MedSAM/official_val/'
incomplete_dir = '/home/qasim/Projects/TurboMedSAM/work_dir/CVPRMedSAMRepViTm11/results_npz/'
search_prefix = '*.npz'

files_in_complete_dir = glob.glob("{}{}".format(complete_dir,search_prefix))
files_in_incomplete_dir = glob.glob("{}{}".format(incomplete_dir,search_prefix))
files_in_incomplete_dir = [os.path.basename(x) for x in files_in_incomplete_dir]

missing_files = [x for x in files_in_complete_dir if os.path.basename(x) not in files_in_incomplete_dir]

print('missing files:', len(missing_files), len(files_in_incomplete_dir), len(files_in_complete_dir))
# for f in missing_files:
#     print(f)
    
missing_files = ['  \"'+x+'\",\n' for x in missing_files if '3D' not in x]
with open('trash.txt', 'w') as writer:
    writer.write('missing_files = [\n')
    writer.writelines(missing_files)
    writer.write(']')
