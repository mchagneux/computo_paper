# Setting up the environment 

```shell
conda env create -f environment.yml
``` 

# Working on the paper 

You can either:
- Work on the .ipynb file which will be synced with the .md file
-  Work directly on the .md file


If you choose the second option, you need to remove `jupytext --sync template-computo-myst.ipynb` from the Makefile.

Run: 

```shell
make
```

Then check the generated paper which is the `.html` file in `build`
