Surrogate Model-Based Explainability Methods for Point Cloud NNs
==============
This work is based on point cloud networks ([this repo](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)) and local surrogate model-based explainability method LIME ([this repo](https://github.com/marcotcr/lime)). Please follow their instructions to build up the environments.

Usage
--------------
After installing LIME, copy the scripts in lime_script/ into lime/ (lime_base will be replaced)

Move modelnet40_test_lime.txt to data/modelnet40_normal_resampled/ or run sample_test_data.py to sample user-defined test set (this py file should also be moved into data/modelnet40_normal_resampled/).

Visualize the explanation for one instance:

    python LIME_single.py

Evaluate a batch of explanations:

    python test_batch_LIME.py

<img src="https://github.com/Explain3D/LIME-3D/blob/main/pic/exp.png?raw=true" width="500px">
