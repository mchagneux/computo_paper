compile:
	jupytext --sync template-computo-myst.ipynb  
	jupyter-book build .
