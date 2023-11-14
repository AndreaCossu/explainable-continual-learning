# Drifting explanations in continual learning

To reproduce the experiments we provided the scripts in the file `scripts.sh`. 

The scripts trains all the models and saves the results in the folder `experiment_folder` provided by command line.
The `script.sh` file does not include the path to the experiment folder, you can add it pointing to the folder of your choice.  
Also, for the SSC experiments you need to provide the path containing the preprocessed dataset, that you can download [here](https://drive.google.com/drive/folders/0B_-e1I5H15TmfjJDc1FPNzlHNmF3dkVQc1FwZGF0SlVPMTI4OVdhanhEUjQ4OEpubGdvNjg?resourcekey=0-mNOj_CNJMlMb62DZBy3NhA&usp=sharing).  
You can select the explanator you want by using either `shap` or `lift` in the command line arguments.  

To get all the plots of the paper, you should run `explanation_over_time.py` by providing the same experiment folder as command line input.  
For each benchmark, continual learning strategy, model and explanator, the script will create different plots for the different metrics.

