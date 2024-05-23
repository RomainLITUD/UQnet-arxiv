# Unravelling uncertainty in trajectory prediction using a non-parametric approach

- Visit the DiTTlab demo page below to visualize how the model works:
### [Dittlab Online Demo (click)](http://mirrors-dev.citg.tudelft.nl:8082/uqnet-main.py/)
- This is the official source code of the paper manuscript: **Unravelling uncertainty in trajectory prediction using a non-parametric approach**

- The paper is available on Transportation Research Part C: Emerging Technologies : [website link (click)](https://www.sciencedirect.com/science/article/pii/S0968090X24001803)

## Quick start

### Requirements:

* Python = 3.9
* PyTorch ≥ 1.11
* Shapely = 1.8.5

### Packages

* Data preparation needs ROS and lanelet2 toolkit, which only supports Linux systems. Please go to [github page](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) to see the installation tutorial

* Before training or testing the model, please install the required packages by:

``` bash
 pip install -r packages.txt
```

### Data Preparation

* The used INTERACTION dataset can be found by [Interaction Webpage](https://interaction-dataset.com/). 
* The corresponding INTERPRET challenge: [Leaderboard](http://challenge.interaction-dataset.com/leader-board).
* Put the INTERACTION data in the correspondings train/val/test folders in `interaction_data` folder
* Use `data_generator.ipynb` and `mask_generator.ipynb` to process the raw data. These two notebooks provide detailed instructions.
* Processed data will be in `interaction_merge` folder.
* If there is any difficulty in preparing the dataset, please first get the data permission from INTERACTION team and send us an email: [G.Li-5@tudelft.nl](G.Li-5@tudelft.nl). We will share the fully-processed data that is ready to use.

### Model Training

* Run the `TrainingModels.ipynb` to train the deep ensembles of UQnet.
* Detailed instructions are provided in the notebook.

### Paper Reproduce

* Run the `ResultsReproduce.ipynb` to get the quantified uncertainty, predictions, etc.
* Here we use the MR minimization strategy, which is the same as the leaderboard.
* Detailed instructions are provided in the notebook.

### Reproducing the Accuracy on the Leaderboard

* Run the `ForSubmission.ipynb` sequentially.
* The generated submission file is exactly the same as shown on the leaderboard

### Visualization

* For visualization, please do not use the `VisualizeResults` notebook. It is the old test.
* Please go to our online interactive demo for visualization (the link is at the top of the page).


![alt text](https://github.com/RomainLITUD/UQnet-arxiv/blob/main/figs/archi.jpg "Model Structure")
