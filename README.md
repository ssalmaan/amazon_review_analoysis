Setup a conda environment using
>> conda env create -f environment.yml

To activate the environment

>> conda activate develop

Run the code using
>> python main.py --analysis_type N
where N is a number from 1 to 6 depending on the task to be run.

Run with the --no-preload flag for the first time to preprocess and save data needed for analysis. Subsequently, use the --preload flag for fast execution.

Also, take a look at the notebook (ipynb or pdf) for an overview of the results.
