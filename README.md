# BBBResearch
Using FCNNs and quadratic polynomial regression to predict permeability of compounds through the Blood-Brain Barrier
The full feature dataset (509 molecules) as well as the actual dataset (281 molecukes) that was fed in for training/testing are both located in the repository. The feature search yielded many "not found" values-- molecules with more than 50% of data incomplete were removed from the dataset.
The code is written entirely in Python.

How to run the code: Any Python IDE can run the code with the right packages (found in requirements.txt)

The following steps are to be followed to install and run the code base.

Requirements:
1. Python3.7 installed in a machine with venv available

2. Install the necessary libraries using the following code inside PIP.  
`pip install -r requirements.txt`

3. Make sure you are now in the same directory as the codebase.

4. To run the demo version of the application with some test sets, please run 
`python predict.py` in the current directory.

5. The codebase will load the `bbb-model ` into the memory and will run the test dataset and will plot evaluation metrics.

6. If you need to run a specific model, please feel free to change the `predict` function to point to the respective feature set of the target molecule.
