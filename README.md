# Setting up the environment 

```shell
conda env create -f environment.yml
``` 

# Working on the paper 

First run: 

```shell 
jupytext --set-formats ipynb,md:myst paper.ipynb
```
You can edit the notebook and run cells in a notebook environment connected to the "computo" kernel. 

Once your notebook is ready with no errors from the code cells, you can build the document with:

```shell
make
```

Then check the generated paper which is the `.html` file in `build`
