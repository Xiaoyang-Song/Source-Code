conda create -n DCCA-AM python=3.9.7
conda activate DCCA-AM

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install tqdm
pip install tensorboard
pip install torchinfo
pip install PyYAML