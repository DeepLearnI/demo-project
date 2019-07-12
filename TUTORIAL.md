<img style="float: left;" src="https://dessa.com/wp-content/uploads/2018/05/dessa_logo.svg" height="50">
<p align="right"> Powered by <img src="https://cloud.google.com/_static/images/cloud/icons/favicons/onecloud/super_cloud.png" height="50" width="50" >
</p>
<br>
<hr>


# Welcome to Foundations

*Estimated time: 20 minutes*

Welcome to the Foundations trial environment! 

In this tutorial we'll go through the process of optimizing and 
serving a simple text generator
model using Foundations.

This trial environment provides you with 
a fully managed Foundations setup, including:


* 10 GPUs 
* Foundations, TensorFlow, and the Python scientific stack 
(NumPy, pandas, etc.) pre-installed 
* An in-browser IDE 


Foundations is infrastructure-agnostic 
and can be set up on-premise 
or on the cloud. It can be used with any development environment.



In this trial we will start by taking some basic model code and using it 
to explore some of Foundations' key features.


This is what we'll achieve today:


1. With minimal effort, we will optimize our model with an 
architecture and hyperparameter
search on a cluster of machines with GPUs on Google Cloud Platform

1. We will track and share metrics to assess model performance

1. We will see how Foundations automatically tracks the parameters and 
results of these experiments in a dashboard, enabling full reproducibility.

1. Finally, we'll select the best model and serve it to a demo web app.  

## Hello, world!

Let's submit a job with Foundations. Run the following command in the terminal:

```bash
$ foundations deploy --env scheduler --job-directory experiments/text_generation_simple
```

Congratulations! The job is running and the model is training remotely on GPUs. 

Any code can be submitted in this way without modifications. 

Now let's scale up our experimentation with Foundations.

## Architecture and parameter search

To the right of this pane, you will see `main.py`. This code was 
quickly assembled by one of our machine learning engineers without using 
Foundations. 

The model is a [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) (gated recurrent unit) language generator. We will train it on 
some Shakespearean text, and the resulting model 
will be able to synthesize new text that sounds 
(ostensibly) like Shakespeare. 
 
We're going to optimize the model performance using an architecture and parameter
 search. 


### Create a job deployment script

 Without Foundations, running a search over many 
 architectures and sets of hyperparameters
 is messy and difficult to manage. Foundations makes this
  straightforward! We're going to 
 write a simple script to immediately kick off 
 20 jobs of a random search on our cluster. 

In the editor, right click on the `experiment_management/` folder 
and create a 
new file called `submit_jobs.py`. Add in the 
following code:

```python
import foundations
import numpy as np

# Constant for the number of models to be submitted 
NUM_JOBS = 20 

# Get params returns randomly generated architecture specifications 
# and hyperparameters in the form of a dictionary
def generate_params():
    params = {
        "rnn_layers": np.random.randint(1, 4),
        "rnn_units": np.random.randint(128, 513),
        "batch_size": np.random.randint(32, 257),
        "embedding_dim": np.random.randint(64, 257),
        "epochs": np.random.randint(3, 11),
        "learning_rate": np.random.choice([1e-3, 5e-3, 1e-2]),
        "temperature": np.random.choice(np.arange(0.1, 1.1, 0.1)),
        "seq_length": 100,
        "dataset_url": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    }
    return params
    
# A loop that calls the deploy method from the  
# Foundations SDK which takes in a parameters dictionary
# and the entrypoint script for our code (main.py)
for _ in range(NUM_JOBS):
    foundations.deploy(
        env="scheduler",
        job_directory="experiments/text_generation_simple",
        entrypoint="main.py",
        project_name="a project name",
        params=generate_params(),
    )
```

### Load parameters from Foundations

Start by adding an import statement to the top of `main.py`:

```python
import foundations
```

Beginning on line 7, the code has a locally defined parameters dictionary.
Replace that with the following line:
```python
params = foundations.load_parameters()
```

### Track metrics

In `main.py` we have lines which print 
useful information about our model. It's easy to get 
Foundations to log them. 

Around line 30, there is the following code:


```python
train_loss = model.test(dataset_train, steps_per_epoch_train)
print("Final train loss: {}".format(train_loss))

test_loss = model.test(dataset_test, steps_per_epoch_test)
print("Final test loss: {}".format(test_loss))

# Change the model to test mode
model.set_test_mode(checkpoint_dir='./training_checkpoints')

# Prompt the model to output text in the desired format
generated_text = model.generate(start_string=u"ROMEO: ", num_characters_to_generate=25)
print("Sample generated text: \n{}".format(generated_text))
 ```
    
To track any performance metric using Foundations, you can 
call `log_metric`. Let's add the following lines to the bottom of `main.py`:
 
 ```python
foundations.log_metric("train loss", train_loss)
foundations.log_metric("test loss", test_loss)
foundations.log_metric("sample output", generated_text)
```

Foundations can track any number or string in any part of your project code this way.


### Prepare model for serving

In order to serve the model later, we'll need to prepare the `predict.py` 
entrypoint and create a configuration file.

Open `predict.py` and add an import statement to the top:

```python
import foundations
```

Also replace the `params` dictionary with

```python
params = foundations.load_parameters()
```


Now we need a configuration file to standardize the model entrypoint for serving.

Right-click on the `text_generation_simple` folder and create a new file called 
`foundations_package_manifest.yaml` and paste the following text into it:

```yaml
entrypoints:
    predict:
        module: predict
        function: generate_prediction
```


### Launch parameter search


At the bottom of this window you'll see a terminal. 
Type the following command
 to launch the script we just wrote:
 

```bash
$ python experiment_management/submit_jobs.py
```

That's it! We're now using the full capacity 
of available compute resources to explore our architecture and 
parameter space by training 20 models 
concurrently. The jobs will now be deployed and run in the background.

Let's take a look at how they're doing.


## Dashboard

Foundations provides a dashboard that allows teams to monitor 
and manage 
multiple projects across a cluster. We can take a look at the 
parameters and performance metrics of all the jobs we 
submitted. 


Click [here](DASHBOARD_URL) to open the dashboard.


| Icon           | Status                   |
|----------------|--------------------------|
|      green     | Job complete             |
| green flashing | Currently running        |
|     yellow     | Queued                   |
|       red      | Job exited with an error |


Some jobs will already be completed. We added a sample
of generated output as a metric — hover 
over a few examples 
to see how our initial models are doing.

---


## Serving

Foundations provides a standard format to seamlessly package machine learning models for production.

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

Foundations automatically retrieves the bundle associated with the job_id 
and wraps it in a REST API. Requests can be made to the entrypoints 
specified by the `foundations_package_manifest.yaml` file.

### Set up pre-baked web app 


Click [here](WEBAPP_URL) to go to a demo webapp that makes a REST call
to the model being served.
For the Model Name, use the IP address 
given in the Slack message. Now try outputting generated text
from your served model!

### Next steps

We want to hear your feedback about Foundations! 

* Fill out this 5-minute [feedback survey](link to Google form)
* Tell us what you thought of Foundations
[via email](mailto:feedback@dessa.com)
* Tweet us [@Dessa](https://twitter.com/dessa) with your best model-generated text using [#FoundationsML](https://twitter.com/search?q=%23FoundationsML)
