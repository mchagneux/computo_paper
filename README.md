
# Template for a Computo submission based on Jupyter-book

[![build output](https://github.com/computorg/template-computo-myst/workflows/build/badge.svg)](https://computorg.github.io/template-computo-myst/)

Documentation and sample of a Myst (Jupyter-book) submission for the Computo journal.

Shows how to automatically setup and build the HTML and PDF outputs, ready to submit to our peer-review platform.

## Process overview

Submissions to Computo require both scientific content (typically equations, codes and figures) and a proof that this content is reproducible. This is achieved via the standard notebook systems available for R, Python and Julia (Jupyter-book and Rmarkdown), coupled with the binder build system. 

A Computo submission is thus a git(hub) repository like this one typically containing 

- the source of the notebook (a .md file + .yml config files + a BibTeX + some statics files typically in `figs/`)
- a configuration file for the binder environment to build the final notebook files in HTML (`environment.yml`). 

The following picture gives an overview of the process on the author's side (with both Rmarkdown/Myst approaches):

![Computo author process](https://computo.sfds.asso.fr/assets/img/computo_process_authors.png)

## Step-by-step procedure

### Step 0: setup a github repository

Clone/copy this repo to use it as a starter for your own contributions.

**Note**: _You can rename the .Rmd and .bib files at your convenience, but we suggest you to keep the name of the config files unchanged, unless you know what you are doing._

Typical git manipulations involve the following commands (change `my_github_account` and `my_article_for_computo`): by doing so, you will keep changes from the computo template if need (optional)

``` bash
git clone https://github.com/computorg/template-computo-Rmarkdown.git
git remote rm origin
git remote add origin https://github.com/my_github_account/my_article_for_computo.git
git remote add upstream https://github.com/computorg/template-computo-Rmarkdown
```

### Step 1. write your contribution 

Write your notebook as usual, as demonstrated in the `template-computo-myst.Rmd` sample. Edit `_config.yml` appropriately.

**Note**: _Make sure that you are able to build your manuscript as a regular notebook on your system before proceeding to the next step._

### Step 2: configure your binder environement

The file `environment.yml` tells binder how to setup the machine used to build your notebook with a conda environment. It must be configured to have all the dependencies required to run you notebook (R, Python, packages/feedstocks and system dependencies).

The default uses conda-forge and includes a couple of popular R packages, with a recent version of `R`:

``` yaml
name: computo
channels:
    - default
dependencies:
    - python
    - pip
    - pip:
        - -r file:requirements.txt 
```

The available feedstocks (Python modules) for conda-forge are listed here: [https://conda-forge.org/feedstock-outputs/index.html](https://conda-forge.org/feedstock-outputs/index.html).

### Step 3: proof reproducibility

It is now time to put everything together and check that your work is indeed reproducible! 

To this end, you need to rely on a github action, whose default is found here: [.github/workflows/build.yml](https://github.com/computorg/template-computo-Rmarkdown/blob/main/.github/workflows/build.yml)

This action will

- Check out repository for Github action on a Mac OS machine
- Set up conda with the Python and R dependencies specified in `environment.yml`
- Render your Rmd file to HTML
- Deploy your HTML on a github page on the gh-page branch

### Step 4. submit

Once step 3 is successful, you should end up with an HTML version published as a gh-page. A PDF file can be obtained by clicking the "Export to PDF" button, which just calls the printing function of your browser (using Chrome should facilitate the rendering of your PDF). This PDF version can be submitted to the [Computo submission platform](https://computo.scholasticahq.com/):

<div id="scholastica-submission-button" style="margin-top: 10px; margin-bottom: 10px;"><a href="https://computo.scholasticahq.com/for-authors" style="outline: none; border: none;"><img style="outline: none; border: none;" src="https://s3.amazonaws.com/docs.scholastica/law-review-submission-button/submit_via_scholastica.png" alt="Submit to Computo"></a></div>
