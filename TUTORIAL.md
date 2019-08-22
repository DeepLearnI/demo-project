<img style="float: left;" src="https://dessa.com/wp-content/uploads/2018/05/dessa_logo.svg" height="50">
<p align="right"> Powered by <img src="https://cloud.google.com/_static/images/cloud/icons/favicons/onecloud/super_cloud.png" height="50" width="50" >
</p>
<br>
<hr>

# Welcome to Foundations Atlas

*Estimated time: 20 minutes*

Welcome to the Foundations Atlas trial environment!

[Foundations Atlas Dashboard](DASHBOARD_URL) (opens in a new tab)

In this tutorial we'll go through the process of optimizing and
serving a fraud detection model using Atlas.

This trial environment provides you with
a fully managed Foundations Atlas setup, including:

* 10 GPUs
* Foundations, TensorFlow, and the Python scientific stack
(NumPy, pandas, etc.) pre-installed
* An in-browser IDE

Foundations Atlas is infrastructure-agnostic and can be set up 
on-premise
or on the cloud. You can submit foundations jobs from any development 
environment; the online IDE you're 
using right now is just an example and Atlas is easy to use from
whatever tools you prefer!

<img src="https://dessa.com/wp-content/uploads/2019/07/f9s_workflow.png">

&nbsp;

The tutorial will demonstrate the following key features of Foundations 
Atlas:

1. We will submit a job using some simple 
code that was not written with Atlas
in mind at all.

1. With minimal modification and effort, we will optimize our model with an
architecture and hyperparameter
search on a cluster of machines with GPUs on Google Cloud Platform

1. We will track and share metrics to assess model performance

1. We will see how Foundations Atlas automatically tracks the parameters and
results of these experiments in a dashboard, enabling full reproducibility.

## Code Overview

The code in `experiments/fraud_demo` is an example of a simple
recurrent model
we will train to 
identify fraudulent credit card transactions. 

The high level directory structure:
```
    experiments/
        fraud_demo/
        text_generation_simple/
        image_segmentation/
    experiment_management/
    requirements.txt
```

We've provided three experiments. This tutorial will walk you through 
turning the first one into a Foundations Atlas project.

The file driver.py is the driver file which runs the pipeline with 
any combination of parameters, trains the model and 
performs inference.

## Hello, world!

Let's submit a job with Foundations Atlas. Run the following command in the terminal:

```bash
$ foundations deploy --env scheduler --entrypoint code/driver.py
```

Congratulations! A job is now running and the model is training remotely on GPUs.

Any code can be deployed in this way without modifications.

## Foundations Atlas Dashboard

Foundations Atlas provides a dashboard that allows teams to monitor
and manage
multiple projects across a cluster. We can take a look at the
parameters and performance metrics of all the jobs we
submitted.

The dashboard can be accessed by entering the following url on the internet browser
```python
https://<cluster name>.dashboard.dessa.com/
```


The dashboard shows a comprehensive list of all the projects being run in the team by different people. For each project, it also shows an interactive list of all the ML experiments and performance metrics.

Each job will show up in the dashboard upon submission, along with an icon indicating the run status.

| Icon           | Status                   |
|----------------|--------------------------|
|      green     | Job completed            |
| green flashing | Currently running        |
|      yellow    | Queued                   |
|       red      | Job exited with an error |

---



## Architecture and hyperparameter search

Now let's scale up our experimentation with Foundations Atlas.

To the right of this pane, you will see a file called `driver.py`. This file drives the processing of training and validating a model. 

The model is a temporal convolutional network (TCN) defined in `model.py`. It will be trained it on a dataset of credit card transactions.  

We're going to optimize the model performance using an architecture and hyperparameter search.

### Create a job deployment script

 Without Foundations Atlas, running a search over many
 architectures and hyperparameters
 is difficult to manage and keep track of. Foundations Atlas makes this
  straightforward! We're going to
 write a simple script to kick off a random search of our hyperparameters.

In the editor, right click on the `code/` folder
and create a
new file called `submit_jobs.py`. Add in the
following code:

```python
import foundations
import os
import numpy as np
import yaml

with open('config/config.yaml') as configfile:
    all_params = yaml.load(configfile)


# Constant for the number of jobs to be submitted
NUM_JOBS = 1


# Generate_params randomly samples hyperparameters to be tested
def sample_hyperparameters(all_params):
    '''
    Randomly sample hyperparameters for tuning.
    :param all_params: all the parameters coming from init_configuration function
    :return: all parameters with randomly sampled hyperparameters
    '''
    all_params['tcn_units'] = [int(np.random.choice([128, 256, 512]))] * 3
    all_params['tcn_dropout'] = [float(np.random.choice([0.05, 0.1, 0.2]))] * 3
    all_params['kernel_sizes'] = [int(np.random.choice([3, 5]))] * 3
    all_params['dropout'] = [float(np.random.choice([0.05, 0.1, 0.2]))] * 6
    all_params['batch_size'] = int(np.random.choice([32, 64, 128, 256]))
    all_params['learning_rate'] = float(np.random.choice([0.001, 0.005, 0.01]))

    return all_params

# A loop that calls the deploy method for different combinations of hyperparameters
for _ in range(NUM_JOBS):
    all_params = sample_hyperparameters(all_params)

    with open(os.path.join('config','hyperparams_config.yaml'), 'w') as outfile:
        yaml.dump(all_params, outfile, default_flow_style=False)

    foundations.deploy(
        env="scheduler",
        job_directory=".",
        entrypoint="code/driver.py",
        project_name="User - Anti money laundering"
    )
```

This code will sample hyperparameters and launch a job with each set of parameters.

### Load parameters from Foundations Atlas

Start by adding an import statement to the top of `driver.py`:

```python
import foundations
```

Around line 14, replace the following line
```python
all_params = init_configuration(config_file='config/config.yaml')
```
with these lines.

```python
all_params = init_configuration(config_file='config/hyperparams_config.yaml')
foundations.log_params(get_arguments_as_dict(all_params))

```
By doing this, a random sample of hyperparams are read from 'hyperparams_config.yaml' (which is generated by submit_job.py for each job). The line foundations.log_params is used to log these params into foundations system so that they can be tracked via GUI.


### Track metrics
Start by adding an import statement to the top of `model.py` since all of the metrics are being calculated in this file:

```python
import foundations
```

In `model.py` we have lines which print
useful information about our model. It's easy to get
Foundations Atlas to log them.

Around line 328, we have the following code:

```python
###### Replace these lines ###########################
print(f'train_loss:  {float(loss)}')
print(f'validation_loss{float(val_loss)}')
print(f'validation_capture_rate{float(capture_rate)}')
######################################################

```

Replace these lines with the following code:
```python
foundations.log_metric('train_loss', loss)
foundations.log_metric('validation_loss', val_loss)
foundations.log_metric('capture_rate', capture_rate)
 ```

Foundations Atlas can track any number or string in any part of your project code this way.

### Save artifacts

During model training, various artifacts may be produced which we'd like to save for later inspection. For instance, tensorboard artifacts.

Below the log_metric lines shown above, add the following lines:

 ```python
foundations.save_artifact(tensorboard_file, key='tensorboard')
foundations.save_artifact(fig_path, key='average precision recall_{}'.format(step))
```

This way, our tensorboard artifacts will be directly accessible from the dashboard.

Around line 408 in model.py, add the following line in order to track the inference speed of the trained model in foundations gui
```python
foundations.log_metric('avg_inference_time(sec)', np.mean(time_list))
```

If you want to track how much TensorRT can prune your neural network to perform faster inference, you can add the following lines around line 514 in model.py.

```python
foundations.log_metric('num_nodes_trained_model', all_nodes_frozen_graph)
foundations.log_metric('num_nodes_TensorRT_model', all_nodes)
```

### Launch Hyperparameter Search

At the bottom of this window there's a terminal.
Type the following command
 to launch the script we just wrote:


```bash
$ python code/submit_jobs.py
```

That's it! Foundations Atlas is now using the full capacity
of available compute resources to explore our architecture and
parameter space by training a group of models
concurrently. To run the jobs, the scheduler puts them in a queue. It
then will automatically spin up GPU machines in the cluster
as needed up to the configured limit, and run jobs until the queue is
empty. When a worker machine is not being used, the scheduler will automatically
spin it down after a short timeout window.  
In this way it maximizes available resources while
minimizing the amount of time instances are left idle.

Let's take a look at how the submitted jobs are doing.


## Select the best models from dashboard

Foundations Atlas dashboard provides various sorting and filtering operations that allow to quickly select only the acceptabel models. For example, if our acceptance criteria is capture_rate > 80 and inference_speed < 1 millisecond, we can quickly do this dashboard as shown below:



## Serving

Foundations Atlas provides a standard format to seamlessly package machine
learning models for production.

We've included a configuration file `foundations_package_manifest.yaml`
which tells Foundations Atlas to serve `generate_prediction(...)` from `predict.py`


We will use the `predict.py` function and package `yaml` created earlier to serve the model.


### Select the best model

On the dashboard, select a job to serve. It is recommended to choose the one
with the lowest `test_loss`, or perhaps your favourite generated
text example. Copy the `job_id`.

### Serve the model

In the terminal, enter

```bash
foundations serve start <JOB_ID>
```

Foundations Atlas automatically retrieves the bundle associated with the job_id
and wraps it in a REST API. Requests can be made to thrile.

## Next steps

What are the next steps? Well...anything! Try using your own data and
writing (or copying) your own model
code. Experiment further in any way you like!

After that, we would like to hear your what you think about Foundations Atlas!

* Fill out this [feedback survey](https://docs.google.com/forms/d/1Zs4vZViKgdsa6_0KwgUNIA5MgVAcN77RrX8YN4kxqfo)
* Tell us what you thought of Foundations Atlas
[via email](mailto:foundations@dessa.com)
* Tweet us [@Dessa](https://twitter.com/dessa) with your
best model-generated text using
[#FoundationsML](https://twitter.com/search?q=%23FoundationsML)
