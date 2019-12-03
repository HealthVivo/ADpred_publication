import os,re,sys
import subprocess
from termcolor import colored
from datetime import datetime

INFO = '\n\nThere are 3 experiments: '+ colored('ARG3, ARG1 and ILV6', 'green') + '. Each contains a presorting group and bins 1-4 sorted groups, where:\n\n' + \
colored('bin1', 'white', 'on_red') + ': NEGATIVE samples. Should not contain ADs --> EXCEPT for the bin1 in the first experiment (ARG3), where bin1 has very weak activators and NOT negative...\n' + \
colored('bin2','white', 'on_red') + ': ~weak\n' + \
colored('bin3','white', 'on_red') + ': ~medium strength\n' + \
colored('bin4','white', 'on_red') + ': ~strong AD\n\
You should call the script: ' + colored('python FLASH_wrapper.py ARG1 (or arg3 of ilv6)', attrs=['bold']) + '\n' + \
'NOTE: A folder with the name of the promoter will be created in s3://fh-pi-hahn-s/results/ directory if a docker image is provided to a future user without access to our s3 bucket, he should change this folder-name in the scripta\n\n'

OUTPUT_PATH = 's3://fh-pi-hahn-s/results/' + datetime.now().strftime("%Y-%m-%d") + '/' 


def main():

    if len(sys.argv)==1:
        print(INFO)
        sys.exit(1)
    
    run_flash_jobs(sys.argv[1])

# These are the absolute paths were the data is storaged. For now these are private to us but the addresses will have to be changed to GEO addresses after this data gets published.
ARG3_PATH = 's3://fh-pi-hahn-s/Activators/data/151203/fastq/' #'/shared/ngs/illumina/aerijman/151203_D00300_0225_AHGJM3BCXX/Unaligned/Project_aerijman/'
ARG1_PATH = 's3://fh-pi-hahn-s/Activators/data/170413/fastq/' #'/shared/ngs/illumina/aerijman/170413_SN367_0904_AHKNKTBCXY/Unaligned/Project_aerijman/'
ILV6_PATH = ARG1_PATH 										  

# The purpose of these disctionaries are to simplify handling weird names to access the data.
ARG3 = {
    'bin1':['Sample_O1-1','Sample_O2-1'],
    'bin2':['Sample_O1-2','Sample_O2-2'],
    'bin3':['Sample_O1-3','Sample_O2-3'],
    'bin4':['Sample_O1-4','Sample_O2-4'],
    'pre_sorting':['Sample_O1-18_8_15','Sample_O2-18_8_15','Sample_O1-7_8_15']
}
ARG1 = {
    'bin1':['Sample_GCTACGC'],
    'bin2':['Sample_CGAGGCT'],
    'bin3':['Sample_AAGAGGC'],
    'bin4':['Sample_GTAGAGG'],
    'pre_sorting':['Sample_CGTACTA']
}
ILV6 = {
    'bin1':['Sample_GGACTCC'],
    'bin2':['Sample_TAGGCAT'],
    'bin3':['Sample_CTCTCTA'],
    'bin4':['Sample_CAGAGAG'],
    'pre_sorting':['Sample_TAAGGCG']
}


def run_flash_jobs(experiment):
	# Paths to each experiment is complicated and folders are named through sequencing indexes. 
	# Hence, I summarize the paths in the dictionaries defined above 
    experiment = experiment.upper()
    _path = '_'.join([experiment, 'PATH'])
    # mkdir if necessary
    if not os.path.isdir(OUTPUT_PATH + experiment): os.mkdir(OUTPUT_PATH + experiment)
    # go to the new folder... This is probaly not optimal, but I can avoid writing a lot of sequentially joined names downstream, avoiding typing errors...
    os.chdir(OUTPUT_PATH + experiment)
    here = os.path.abspath('./')

    # copy and gunzip all files. Mkdir if necessary 
    print('... Copying and gunziping...')
    for k,v in eval(experiment).items():
        if not os.path.isdir(k): os.mkdir(k)
        for i in v:
            copy_files = 'cp {}{}/*fastq.gz ./{}/'.format(eval(_path),i,k)
            os.system(copy_files)
            gunzip_files = 'gunzip ./{}/*gz'.format(k)
            os.system(gunzip_files)

	# run FLASH on all files. ChDir-ing all along
    print('... running FLASH...')
    for k,v in eval(experiment).items():
        this_path = os.path.join(here,k)
        os.chdir(this_path)
        for f1 in [f for f in os.listdir(this_path) if re.search("_R1_",f)]:
            f2 = re.sub('_R1_','_R2_', f1)
            out = re.sub('_R1_','_R_', f1)
            flash = '~/aerijman/FLASH-1.2.11/flash {} {} -m 3 -M 99 -o {} >> results.log'.format(f1,f2,out[:-6])
            os.system(flash)
            # ~/FLASH-1.2.11/flash R1 R2 -m 4 -M 7 -r 100 -f 193 -s 3 > fastas	
    # combine files into a single file.
    for k,v in eval(experiment).items():
        this_path = os.path.join(here,k)
        os.chdir(this_path)
        files = ' '.join([i for i in os.listdir(this_path) if re.search("extendedFrags",i)])
        single_file = ''.join([k,'.fastq'])
        os.system('cat {} > {}'.format(files, single_file))


if __name__=='__main__': main()


