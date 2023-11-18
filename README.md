The unique folding pattern of the human cortex presents challenges in accurately detecting and annotating sulcal features from brain MRI images, due to the complexity and inter-individual variability
of cortical folding. While primary sulci can be identified algorithmically, detecting secondary sulci is more difficult due to their high variability in
form and presence. To address this, we utilized two datasets and four input modalities (sulcal skeleton and white/grey surface of the left and right 
hemisphere) to classify the existence or absence of the paracingulate sulcus (PCS) using different 3D Convolutional Neural Network (CNN) deep learning networks. To ensure correct training, we employed two 3D local explainable techniques (3D GradCam and 3D SHAP)
and generalized the results using Principal Component Analysis (PCA) to study the global learning patterns of the networks.

# Install python code

cd unzip_fold_of_code/

pip install .

# Source code description

Please if you use the code cite the above publication:

[1]
