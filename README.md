# Tac-VGNN

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

This is the official implementation of [Tac-VGNN: A Voronoi Graph Neural Network for Pose-Based Tactile Servoing](https://sites.google.com/view/tac-vgnn/home).


By [Wen Fan], [Max Yang], [Yifan Xing], [Nathan Lepora](https://scholar.google.com/citations?hl=zh-CN&user=fb2WiJgAAAAJ), and [Dandan Zhang](https://scholar.google.com/citations?hl=zh-CN&user=233I39oAAAAJ).



![](https://github.com/Neo-manchester/Tac-VGNN/blob/main/README_IMG/voronoi_graph_generation.png)
![](https://github.com/Neo-manchester/Tac-VGNN/blob/main/README_IMG/vgnn_interpretability_crop.png)

 Code will be released soon after ICRA 2023.
 
 # Installation
 
 Code was developed and run from Visual Studio Code in Anaconda virtual environment on Windows 11. To install this package, clone the repository into your own folder firstly:
 
 ```
 git clone https://github.com/Neo-manchester/Tac-VGNN.git
 ```

The package structure is shown as:

```
Tac-VGNN                         
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





## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/Neo-manchester/Tac-VGNN/blob/main/LICENSE) file.

