# Bioinformatic analysis for the manuscript Erijman et al., 2019

## Installation. 
1. [install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) (unless you already have it): 
2. clone the repository in your computer: `git clone git@github.com:aerijman/ADpred_publication.git` 
3. Build dependencies.   
  __If you work with conda:__  
  * [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html). 
  * In a terminal window:   
    * `conda create -n adpred python=3.7.4`. 
    * `conda activate adpred`. 
    * `while read requirement; do conda install --yes $requirement; done < dependencies.txt`.     
 
  __If you work with pip:__   
  * You should have `python>=3.6.5` otherwise, install it.   
  * run `pip install -r requirements.txt`.      
4. run the notebook: `jupyter notebook analysis.ipynb` (I prefer jupyter lab)
---  

## Analysis
The first cell of the jupyter notebook downloads the data from its [Dropbox address](https://www.dropbox.com/s/vooe7mb62rnswp5/data2.tar.gz?dl=0).   
_Alternatively_, You could download the data outside the notebook `wget https://www.dropbox.com/s/vooe7mb62rnswp5/data2.tar.gz?dl=0` and start the notebook from cell 2.  






## Preprocessig consisted in:
1. Constructing the complete insert from the reads with `FLASH_wrapper.py` which wraps [FLASH tool](https://ccb.jhu.edu/software/FLASH/)
2. Translating the nucletide sequences into protein sequences with `translate.py`	
3. Clustering similar sequences to reduce noise/variation from sequencing and dna handling errors with `run_usearch.sh`	which wraps [usearch tool](usearch) and `clusters.py`	
4. Secondary structure and disorder predictions were automatized with external_software.sh and use [psipred](http://bioinf.cs.ucl.ac.uk/psipred/) and [iupred](https://iupred2a.elte.hu)
