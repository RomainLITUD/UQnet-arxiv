# UQnet

- Visit the DiTTlab demo page below to visualize how the model works:
### [Dittlab Online Demo (click)](http://mirrors-dev.citg.tudelft.nl:8082/uqnet-main.py/)
- This is the official source code of the paper manuscript: **<UQnet: quantifying spatial uncertainty in trajectory prediction by a non-parametric and generalizable approach>**

- The manuscript is under review and preprinted on SSRN: [manuscript (click)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4241523)

## Quick start

Data Preparation

* The used INTERACTION dataset can be found by [Interaction Webpage](https://interaction-dataset.com/). 
* The corresponding INTERPRET challenge: [Leaderboard](http://challenge.interaction-dataset.com/leader-board).
* Put the data in the correspondings folders in `interaction_data` folder
* Use `data_generator.ipynb` and `mask_generator.ipynb` to process the raw data. 
* Processed data will be in `interaction_merge` folder.
* If there is any difficulty on preparing dataset, please first get the data permission from INTERACTION team and send us an email: [G.Li-5@tudelft.nl](G.Li-5@tudelft.nl). We will share the fully-processed data that is ready to use.

Requirements:

* Python = 3.9
* PyTorch ≥ 1.11

### 1) Packages

* Data preparation needs lanelet2 toolkit, which only supports Linux systems. Please go to [github page](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) to see the installation tutorial

* Before training or testing the UQnet, please install the required packages by:

``` bash
 pip install -r packages.txt
```


![alt text](https://github.com/RomainLITUD/UQnet-arxiv/blob/main/figs/archi.jpg "Model Structure")
