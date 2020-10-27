# epidemiological-gcn

This repository contains the source code of the paper *"An Epidemiological Neural network exploiting Dynamic Graph Structured Data applied to the COVID-19 outbreak"*. 

Since the private nature of data used in our work, we cannot share data and experiments. The files *model.py* and *config.json* contain modules which can be easily plug in [this Pythorch template](https://github.com/victoresque/pytorch-template) to analyze your own data. Note that our model is designed to work with the library [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric) for the representation and analysis of graph data structures.

## Model Architecture in a Nutshell
<p align="center">
  <img src="https://i.ibb.co/HYmMzbj/highlevelarch.png" alt="Model Architecture"/>
</p>

Our model takes in input a sequence of graphs representing places interconnected by movements between them. For each place and for each time step, we have one or more labels representing the ground truth of the epidemiological model which has to be applied to predict the next labels. Through the use of GCNs and LSTMs, the *Contact Rate Estimator* analyzes the sequence of changes in movements between places until the current time step, aiming to tune the *contact rate* parameter of the epidemiological model (e.g. SIR, SIRD) applied on top of it. Though designed to analyze movements data, our architecture can be successfully used to tune other epidemiological parameters, such as *recovery rate*, if the proper features are used.

## Citation 
Please acknowledge the following work in papers or derivative software:

    @article{EpiGCN2020Covid,
      author = {La Gatta, Valerio and Moscato, Vincenzo and Postiglione, Marco and Sperlì, Giancarlo},
      title = "{An Epidemiological Neural network exploiting Dynamic Graph Structured Data applied to the COVID-19 outbreak}",
      journal = {IEEE Transactions on Big Data},
      year = {2020},
      issn = {2332-7790},
      doi = {10.1109/TBDATA.2020.3032755},
    }
