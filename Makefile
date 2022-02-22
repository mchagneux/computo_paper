compile:
	jupytext --sync paper.ipynb  
	jupyter-book build .
