# oja_variance
Analysis of the variances of the weights in Hebbian learning with Oja's rule. 
For being able to use this repository please create a virtual environment with all requirements installed.

```
python -m venv venv
pip install -r requirements.txt
```

Once you have the environment you can run ```python oja.py``` to create JSON files with the variance values for different correlations and learning rates saved.

You can turn these JSON files into a single plot saved as a PNG file using ```python visualize.py```.
