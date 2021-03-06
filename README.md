<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/platform-Windows%20%3F-red">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
</a>

## Bioinformatic analysis for the manuscript:   
### _"A high-throughput screen for transcription activation domains reveals their sequence characteristics and permits reliable prediction by deep learning"_

## Installation   
1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) (unless you already have it): 
2. Clone the repository in your computer (__`git clone git@github.com:aerijman/ADpred_publication.git && cd ADpred_publication`__)
3. Build dependencies.   

   __With conda:__  
     * [`install conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html). 
     * In a terminal window, copy-paste:   
       * __`conda create -n adpred python=3.7.4`__. 
       * __`conda activate adpred`__. 
       * __`while read requirement; do conda install --yes $requirement; done < dependencies.txt`__.     

   __With pip:__   
     * [`install pip`](https://pip.pypa.io/en/stable/installing/). 
     * run __`pip install -r requirements.txt`__ (You should have `python>=3.6.5`).       
       
5. __`$(which pip) install -e git+https://github.com/marcoancona/DeepExplain.git#egg=deepexplain`__
6. run the notebook: __`jupyter notebook analysis.ipynb`__ (I prefer jupyter lab)  
---  

## Analysis
- The first cell of the jupyter notebook downloads the data from its [Dropbox address](https://www.dropbox.com/s/vooe7mb62rnswp5/data2.tar.gz?dl=0).   
_Alternatively_, You could download the data outside the notebook `wget https://www.dropbox.com/s/vooe7mb62rnswp5/data2.tar.gz?dl=0` and start the notebook from cell 2.    

- Figures are created running the scripts from the notebook.
   Many of the scripts are very slow. Hence, some of the processes includeed in the notebook have been modified and executed in a high performance cluster at the Fred Hutchinson Cancer Research Center (in some cases with GPUs). All scripts adapted for HPC can be provided upon request to aerijman@fredhutch.org or aerijman@neb.com.  
---  

<br>

### Preprocessig (already done):   
Preprocessing is very time consuming and consisted of 1- pairing the two reads in fastq files; 2- filtering for reads with correct number of bases and without artifacts (e.g. internal stop codons); 3- translation into amino-acids; 4- clustering into similar sequences (some sequences differ in 1 or 2 aminoacids from their parental sequence due to errors during library-preparation and sequencing. We aleviate these sequence divergence in this step); 4- predict secondary structure and disorder from the amino-acid sequences. 
1. Constructing the complete insert from the reads with `FLASH_wrapper.py` which wraps [FLASH tool](https://ccb.jhu.edu/software/FLASH/)
2. Translating the nucletide sequences into protein sequences with `translate.py`	
3. Clustering similar sequences to reduce noise/variation from sequencing and dna handling errors with `run_usearch.sh`	which wraps [usearch tool](usearch) and `clusters.py`	
4. Secondary structure and disorder predictions were automatized with external_software.sh and use [psipred](http://bioinf.cs.ucl.ac.uk/psipred/) and [iupred](https://iupred2a.elte.hu)



ToDo LiSt:
