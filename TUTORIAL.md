![Dessa Logo](https://dessa.com/wp-content/uploads/2018/05/dessa_logo.svg)

# Welcome to Foundations

*Estimated time: 20 minutes*

Welcome to the Foundations trial environment! This trial environment provides you with a fully managed Foundations setup including:

* ??? GPUs
* Foundations, TensorFlow, and Python scientific stack pre-installed 
* An in-browser IDE

Keep in mind Foundations is infrastructure-agnostic and can be set up on premise or on cloud depending on your needs.

In this trial we will start by taking a basic recurrent language model and using it to explore some of Foundations' most important features.

This is what we'll achieve today:

1. Without any Foundations-specific code at all, we will submit our code to be executed on a remote machine with a GPU

1. We will then add our first lines of Foundations code to store metrics of our choosing

1. We will try to optimise our model using Foundations' hyperparameter search feature. We will see how Foundations can execute multiple jobs using all available computation resources, and can track all parameters and results in an experiment log, allowing for full reproducibility.

1. While doing this we will see how you can use Foundations to manage large scale experimentation.

1. Finally, we'll select the best model and show how easy it is to serve it. 



## 1 Hello, world!

To the right of this pane, you will see `driver.py`. This is a piece of code that was quickly written by one of our machine learning engineers _without using Foundations_. 

The model is a language generator [TODO Mohammed doesn't like this]. We train it on some Shakespearean text so that we're able to synthesize new text that sounds like Shakespeare. 
 
Note that this code is nothing special right now; just some basic Python libraries and TensorFlow. We don't really have to modify it at all in the beginning to just run it on a cluster of GPU machines using Foundations. 


### 1.1 Submit job

Using this one command, we're going to take the code and run it on a remote server with a GPU, all with this one command!

Underneath you'll see a terminal. Go ahead and copy this command into the terminal, setting your own `project-name` if you like:

```
$ foundations deploy --env scheduler --project-name testme --job-directory experiments/text_generation_simple/src --entrypoint driver.py --num-gpus 0
```

Let's break down this command briefly: 

* `foundations deploy` submits a job
* `--env` looks for a configuration file by that name. In a long-term project you might have need to submit jobs to different environments, or want a configuration file for local submissions. [TODO clarify & expand]
* `--project-name` is a user-specified project title. You can freely create new ones whenever you need
* [TODO]

### 1.2 Look at GUI

In a new tab, open [https://35.231.226.217:6443/projects](https://35.231.226.217:6443/projects). Click the Project you're submitting jobs to.

Note that we haven't tracked any metrics or submitted jobs with lots of different parameters yet, but later we'll be able to easily track them here. 


### 1.2 See latest logs

[TODO there should be a friendlier way, otherwise we'll show it later]

Copy the job_id from the job you just submitted.

Now in the terminal type

```
$ foundations retrieve logs --env scheduler --job_id=<copied_job_id>
```

You can do this at any time while your jobs is running to print their latest terminal output. This is also useful for investigating failed jobs. 


### Receive Slack message

[TODO}]  -- either remove or reformulate depending on whether Slack feature exists

## Experiment queueing management

Tracking experiments is powerful if you do [TODO]. We can track all the hyperparameters we try, and all metrics you can calculate in code about any particular experiment.

[TODO]

### Log a metric 

In `driver.py` we have a couple of lines that collect useful information about our model.  

For now, let's track the train loss and test loss metrics we already have code to calculate.

Start by adding an import statement at the top of `driver.py`:

```
import foundations
```

Look for the following lines


```python
train_loss = model.test(dataset_train, steps_per_epoch_train)
print(train_loss)

test_loss = model.test(dataset_test, steps_per_epoch_test)
print(test_loss)

generated_text = model.generate_text(start_string=u"ROMEO: ", checkpoint_dir='./training_checkpoints', temperature=params['temperature'])
print(generated_text)
 ```
    
 All you need to do to track a metric using foundations is to call `log_metric` on any number or string you want foundations to save
 
 ```python
foundations.log_metric("train loss", train_loss)
foundations.log_metric("test loss", test_loss)
foundations.log_metric("sample output", generated_text[:100])
```



### Explore parameter and architecture space 

In `utils.py` right now, we have a `params` dictionary. This is just a configuration of parameters for convenience so far. What we want is for Foundations to track our parameters for us! 

Start by creating a new file [TODO how] in the `experiment_management/`, and copy the following code in:

```python
import foundations
import numpy as np

def get_params():
    params = {
        "rnn_units": np.random.randint(256, 2049),
        "batch_size": np.random.randint(16, 256),
        "embedding_dim": np.random.randint(128, 512),
        "epochs": 30,
        "seq_length": 100,
        "temperature": 1.,
        "dataset_url": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    }
    return params
    
for _ in range(5):
    foundations.set_job_resources(num_gpus=0)
    foundations.deploy(
        env="scheduler",
        job_directory="experiments/text_generation_simple",
        entrypoint="src/driver.py",
        params=gen_params(),
    )
```

Now in `driver.py`, delete line ???[TODO]

```python
from utils import params
```

And add the following line after `import foundations`:

```python
params = foundations.load_parameters()
```

Now everytime you submit a job, foundations will generate a set of hyperparameters for you, and all hyperparameters will be tracked in the GUI.

Let's try it out! 



## Serving

### Pick a best performing model

### Make code "serveable"

### Ask Foundations to serve it [TODO doesn't exist yet]

### Set up pre-baked web app [TODO doesn't exist yet]


## Load a large model from someone else, use as pretrained model for some new data (GPT)

ALL OF THIS IS STILL TODO

### v1 code, retrieve pre-trained job

### retrain (hopefully)

### serve

