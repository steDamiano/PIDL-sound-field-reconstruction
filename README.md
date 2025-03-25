# A Zero-Shot Physics-Informed Dictionary Learning Approach for Sound Field Reconstruction

This repository contains the code associated with the paper 

>S. Damiano, F. Miotello, M. Pezzoli, A. Bernardini, F. Antonacci, A. Sarti and T. van Waterchoot, "A Zero-Shot Physics-Informed Dictionary Learning Approach for Sound Field Reconstruction", in *Proc. 50th Int. Conf. Acoust. Speech Signal Process. (ICASSP)*, Hyderabad, India, Apr. 2025.

Find the paper here: https://arxiv.org/abs/2412.18348

## How to use this repository

#### Data preparation
- Create a ```data``` directory at root level;
- Download into this directory the published dataset https://data.dtu.dk/articles/dataset/Acoustic_frequency_responses_in_a_conventional_classroom/13315286 


#### Installation
The package relies on a poetry virtual environment (poetry installation instructions can be found here: https://python-poetry.org/docs/). To install the package, run: 
- ```poetry install```: installs the environment with all dependencies

#### Running the code
- To run the experiments using the PIDL method, run the script ```multifrequency_helmholtz_dl.py```
- To run the experiments using the baseline dictionary BL, run the script ```multifrequency_sinc_bl.py```
- To create NMSE surface plots, create the directory ```figures``` at root level and uncomment the plotting lines at the end of two scripts.

#### External references
The code that implements the OLDL method is available at https://github.com/manvhah/local_soundfield_reconstruction.

### Acknowledgements
This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 956962, from KU Leuven internal funds C3/23/056, and from FWO Research Project G0A0424N. This paper reflects only the authors’ views and the Union is not liable for any use that may be made of the contained information. 

This work was supported by the Italian Ministry of University and Research (MUR) under the National Recovery and Resilience Plan (NRRP), and by the European Union (EU) under the NextGenerationEU project.
