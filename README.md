# Tac-VGNN

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

This is the official implementation of [Tac-VGNN: A Voronoi Graph Neural Network for Pose-Based Tactile Servoing](https://sites.google.com/view/tac-vgnn/home).


By [Wen Fan], [Max Yang], [Yifan Xing], [Nathan Lepora](https://scholar.google.com/citations?hl=zh-CN&user=fb2WiJgAAAAJ), and [Dandan Zhang](https://scholar.google.com/citations?hl=zh-CN&user=233I39oAAAAJ).



![](https://github.com/Neo-manchester/Tac-VGNN/blob/main/README_IMG/voronoi_graph_generation.png)
![](https://github.com/Neo-manchester/Tac-VGNN/blob/main/README_IMG/vgnn_interpretability_crop.png)

 Code will be released soon after ICRA 2023.
 
 # Download
 
 Code was developed and run from Visual Studio Code in Anaconda virtual environment on Windows 11. 
 
 To install this package, clone the repository into your own folder firstly:
 
 ```
 git clone https://github.com/Neo-manchester/Tac-VGNN.git
 ```

After then, the main parts of download package structure are shown as below:

```
Tac-VGNN   
|
|—— 1_parameter_setup                                 
|   |—— tactip_127_graph_voronoi_setup.ipynb 
|   |—— tactip_331_graph_voronoi_setup.ipynb
|
|—— 2_voronoi_graph_generation
|   |—— voronoi_graph_generation.ipynb
|
|—— 3_model_evaluation
|   |—— gnn_voronoi_train.ipynb
|   |—— gnn_voronoi_test.ipynb
|
|—— lib
|   |—— blob_extraction.py
|   |—— graph_generate.py
|   |—— voronoi_generate.py
|   |—— pytorchtools.py
|   |—— net.py
|
|—— torch_geometric_whls
|
|—— result
|   |—— train
|   |—— test
|
|—— data
    |—— 127
    |—— 331
```

<details><summary> __WHAT DO THOSE PARTS WORK FOR?__ </summary>
<p>

* 1_parameter_setup/tactip_(127/331)_graph_voronoi_setup.ipynb ：detailed examples to show how parameters tuned. 
 
* 2_voronoi_graph_generation/voronoi_graph_generation.ipynb ：generation tutorial of voronoi graph dataset. 
 
* 3_model_evaluation/gnn_voronoi_(train/test).ipynb ：train and evaluation tutorials of Tac-VGNN model.
 
* lib ：function libraries used for upper three steps.
 
* torch_geometric_whls ：four whl files supporting for torch_geometric running.
 
* result/(train/test) ：folders for train/test materials, including train/val set, test set, best_train_model and plots.
 
* data/(127/331) ：raw image data for graph_voronoi_setup.ipynb and voronoi_graph_generation.ipynb.

</p>
</details>



# Installation

<details><summary> __Needed Dependences__ </summary>
<p>

```
python==3.8.0
numpy==1.24.1
scipy==1.10.1
torch==1.12.1
pandas==1.5.3
pytorch==1.12.0
torchvision==0.13.0
torchaudio==0.12.0
cudatoolkit==11.3.1
ipykernel==6.20.2
matplotlib==3.6.3
opencv_python==4.7.0.68
torch-geometric==2.2.0
torch-cluster==1.6.0
torch-scatter==2.1.0
torch-sparse==0.6.15
torch-spline-conv==1.2.1
```

</p>
</details>



First step, configure anaconda environment:

```shell
#create anaconda env
conda create -n Tac-VGNN python==3.8
conda create Tac-VGNN

#change to your package location
cd your_download_dir

#run requirements.txt
pip install -r requirements.txt
```

Second step, install pytorch seperately through conda:

```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

<details><summary> __NOTE!!!__ </summary>
<p>
 
* Only install pytorch from official site https://pytorch.org/ 

* To fit your own OS and cuda, previous pytorch version found here https://pytorch.org/get-started/previous-versions/
 
* DO NOT use 'pip' to install pytorch instead of 'conda' to prevent negative influence for torch_geometric!!! 
 
   ref: https://stackoverflow.com/questions/73046416/torch-geometric-error-filenotfound-could-not-find-module-conda-envs

</p>
</details>

Third step, install torch_geomatric and four whls:

```shell
#change to torch_geometric_whls folder
cd torch_geometric_whls

#install four wheels seperately
pip install torch_cluster-1.6.0+pt112cu113-cp38-cp38-win_amd64.whl

pip install torch_scatter-2.1.0+pt112cu113-cp38-cp38-win_amd64.whl

pip install torch_sparse-0.6.15+pt112cu113-cp38-cp38-win_amd64.whl

pip install torch_spline_conv-1.2.1+pt112cu113-cp38-cp38-win_amd64.whl

#install torch_geometric in the end
pip install torch-geometric==2.2.0
```

<details><summary> __NOTE!!!__ </summary>
<p>
 
* Prefer to install torch_geometric from official site https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html 

* To fit your own OS and cuda, other torch_geometric whls found here https://pytorch-geometric.com/whl/
 
* DO NOT install torch_geometric BEFORE whls to prevent negative influence for torch_geometric!!! 
 

</p>
</details>




# Workflow

* Open the whole package in VSCode
* Run `1_parameter_setup/tactip_(127/331)_graph_voronoi_setup.ipynb` to learn about how voronoi graph generated from
* Run `2_voronoi_graph_generation/voronoi_graph_generation.ipynb` to build train/val and test datasets based on raw image data
* Run `3_model_evaluation/gnn_voronoi_train.ipynb` to train the Tac-VGNN model and save the best trained model
* Run `3_model_evaluation/gnn_voronoi_test.ipynb` to evaluate the trained Tac-VGNN model and plot the results

<details><summary> __NOTE!!!__ </summary>
<p>
 
* Parameters used in `2_voronoi_graph_generation/voronoi_graph_generation.ipynb` should be pretuned in `1_parameter_setup/tactip_(127/331)_graph_voronoi_setup.ipynb`
 
* Unsuitable parameters can cause voronoi graph generation to fail, adjust them a few more times helps to get a better data set
 
* Only `data/331` (round layout) is used in graph generation, model train and test process, but it is easy to be transferred to the `data/127` (hexagonal layout), by applying different functions provided in `lib/graph_generate.py` and `lib/voronoi_generate.py`
 

</p>
</details>

# Supplementary

Two pretrained models and datasets have been uploaded for those who lacks for interest on self-dataset generation or self-model training.

Google Drive Link: https://drive.google.com/drive/folders/1kSv04t3OuxpWKbccjObIN0S_cHObJkqa?usp=sharing
```
Tac-VGNN Supplementary Materials/pretrained_model 
|
|—— result_1                                 
|   |—— train
|   |—— test
|
|—— result_2                                 
    |—— train
    |—— test
```
<details><summary> __HOW TO USE?__ </summary>
<p>
 
* Replace `result_(1/2)` folder to your own `result` folder and rename it as `result` again 

* Then run `3_model_evaluation/gnn_voronoi_test.ipynb` to evaluate the model performance
 
* `result_1` and `result_2` are two different datasets
 

</p>
</details>


# Servoing

The details about how to collect raw image data and achieve tactile servoing, please look at https://github.com/nlepora/pbts-2d#readme

In brief, replace the CNN prediction part of tactile servoing 2D into the Tac-VGNN part introduced in this repository




## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/Neo-manchester/Tac-VGNN/blob/main/LICENSE) file.

