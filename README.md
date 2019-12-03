## ADpred_publication
* scripts and notebook for publication of ADpred.  
* This repo will be ready later today 12/2/2019

### Installation. 

1. [install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) (unless you already have it): 
2. clone the repository in your computer: `git clone git@github.com:aerijman/ADpred_publication.git` 
3. Build dependencies.   
  __If you work with conda:__  
  * [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html). 
  * In a terminal window:   
    * `conda create -n adpred`. 
    * `conda activate adpred`. 
    * `while read requirement; do conda install --yes $requirement; done < dependencies.txt`.     
 
  __If you work with pip:__   
  * You should have `python>=3.6.5` otherwise, install it.   
  * run `pip install -r requirements.txt`.    

The first cell of the jupyter notebook downloads the data from its [Dropbox address](https://www.dropbox.com/s/vooe7mb62rnswp5/data2.tar.gz?dl=0).   
_Alternatively_, You could download the data outside the notebook `wget https://www.dropbox.com/s/vooe7mb62rnswp5/data2.tar.gz?dl=0` and start the notebook from cell 2.  
