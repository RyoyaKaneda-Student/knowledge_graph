ナレッジグラフ(Knowledge graph Challenge)
==============================

This is the project of Knowledge Graph, especially Knowledge Graph Challenge.
If you have any questions, feel free to ask me!  

name: Ryoya kaneda  
mail: kaneda@ss.cs.osakafu-u.ac.jp 

## How to run training with docker and gpu.

Please change some parts depending on your environment.


1. Clone these repositories.
2. Change directory. ex)`cd knowledge_graph`
3. bash run `docker build -t ${image-name} -f ./docker/Dockerfile .`. 
If you use linux with no gpu, this may be not work, so you change `pyproject.toml` and `Dockerfile`. 
Additionally, I do not have a Windows environment, so if you are in that environment, please make changes accordingly.
4. docker run with these folders. ex)`docker run --rm -it --gpus all -v $PWD:/var/www/ ${image-name} bash`
5. run `makers init_folder` and `makers init_kgc_folder`.
6. run `python3 src/data/make_kgcdata.py`
7. run `python3 src/data/make_missing_kgcdata.py`
8. run `python3 src/run_for_KGC.py *args... `. 
If you want to check args parameters, run `python3 src/run_for_KGC.py -h`

Project Organization
------------

    ├── LICENSE
    ├── Makefile.toml      <- Makefile.toml with commands like `makers ***`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── saved_models       <- Trained and serialized models, model predictions, or model summaries
    ├── log                <- log direction.
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions.
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    ├── docker             <- Dockerfiles. If you don't use GPU, you change it to not use the GPU.
    └── pyproject.toml     <- poetry file. If you don't use GPU, you change it to not use the GPU.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
