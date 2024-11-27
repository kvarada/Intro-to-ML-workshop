# Introductory Machine Learning Full-Day Workshop for Students with programming background

This repository hosts materials for the [CS Alumni Workshop: Is Machine Learning Suitable for your Projects](https://my.cs.ubc.ca/event/2024/12/cs-alumni-workshop-machine-learning-suitable-your). This workshop is a collaboration between [the Master of Data Science, Vancouver program](https://masterdatascience.ubc.ca/programs/vancouver) and [the Department of Computer Science](https://www.cs.ubc.ca/) at UBC. 
 

The module slides are available under: `website/slides/`. This website is built using [Quarto](https://quarto.org/) and you first need to install it if you want to build it locally.   

Here are the steps if you want to build and render the website locally. 

1. Clone this repository and navigate to the repository folder.  
2. Create `conda` environment for the workshop using the `environment.yml` file

```
conda env create -f environment.yml
```

3. Once the environment is created successfully, activate the environment

```
conda activate ml-workshop
```
4. Make the necessary changes in your slides which are located at `Intro-to-ML-workshop-2024/website/slides/`.
5. Once you are ready, navigate to `Intro-to-ML-workshop-2024/website` and render quarto website
```
quarto render 
```
