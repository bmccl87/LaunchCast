# LaunchCast

The model.keras file is the TensorFlow v2.15.0 based model for deployment. It can be found here:
https://github.com/bmccl87/LaunchCast/releases/tag/LC_Model

The test_data.nc file is an example set of the data used as input. It is congruent with the model prediction made in Figure 5 in the paper. 

To run the model, please look at the python file for code to run the model. The file must be run in the same directory ad the LC_model_64216.py file, to allow the tensorflow inherited classes to be imported during the load call.  

If the model.keras file does not load, it is an issue with the tensorflow environment. 

The python environment (pip) requirements is in the requirements.txt file. 
