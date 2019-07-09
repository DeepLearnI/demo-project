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

To the right of this pane, you will see `main.py`. This is a piece of code that was quickly written by one of our machine learning engineers _without using Foundations_. 

The model is a language generator [TODO Mohammed doesn't like this]. We train it on some Shakespearean text so that we're able to synthesize new text that sounds like Shakespeare. 
 
Note that this code is nothing special right now; just some basic Python libraries and TensorFlow. We don't really have to modify it at all in the beginning to just run it on a cluster of GPU machines using Foundations. 


### 1.1 Submit job

Using this one command, we're going to take the code and run it on a remote server with a GPU, all with this one command!

Underneath you'll see a terminal. 

Start by `cd`ing into the project directory:

```bash
$ cd experiments/text_generation_simple
```

Go ahead and copy this command into the terminal, adding your 
own `--project-name` if you like:

```bash
$ foundations deploy --env scheduler 
```

Submitting a job like this will stream the output live; 
you can stop it at any time using Ctrl+C, the job will continue running.

Let's break down this command briefly: 

* `foundations deploy` submits a job
* `--env` looks for a configuration file by that name. In a long-term 
project you might have need to submit jobs to different environments, 
or want a configuration file for local submissions. [TODO clarify & expand]

We are making use of defaults for a few others:

* `--project-name` is a user-specified project title. It defaults to the 
directory name. You can freely create new ones whenever you need
* `--entrypoint` is the name of the script to run, it defaults 
to `main.py` but can be specified manually
* [TODO]

### 1.2 Look at GUI

In a new tab, open [https://<IP>:<PORT>/projects](https://<IP>:<PORT>/projects). Click the Project you're submitting jobs to.

Note that we haven't tracked any metrics or submitted jobs with lots of 
different parameters yet, but later in this tutorial we'll easily track them here. 


### 1.2 See latest logs

[TODO there should be a friendlier way, otherwise we'll show it later]

Copy the job_id from the job you just submitted.

Now in the terminal type

```
$ foundations retrieve logs --env scheduler --job_id <copied_job_id>
```

You can do this at any time while your jobs are running to print their 
latest terminal output. This is also useful for investigating failed jobs. 


### Receive Slack message

[TODO] 

## Experiment queueing management

Tracking experiments is powerful if you do [TODO]. We can track all the 
parameters and architectures 
we try, and all metrics you can calculate in code about any particular experiment.

[TODO]

### Log a metric 

In `main.py` we already have a couple of lines that print 
useful information about our model. It's easy to get Foundations to log them. 

For now, let's track the train loss and test loss metrics we already have. [TODO rewrite]

Start by adding an import statement to `main.py`:

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
foundations.log_metric("sample output", generated_text[:20])
```

Now submit a job again

```bash
$ foundations deploy --env scheduler 
```

You can log any number or string as a metric, anywhere in your code. For example, 
you can add metrics such as ROC AUC or accuracy or a domain-specific business 
metric. Metrics get posted to the job page
as soon as they're recorded, so you can do things like have a Keras callback save
a metric within the training loop. 


Let's go back to the jobs page: [https://<IP>:<PORT>/projects](https://<IP>:<PORT>/projects)

You should be able to see the metrics you've saved in the right pane.

### Submit jobs via script

In addition to submitting jobs via the command line, it's straightforward to submit
them within python code as well.

Start by creating a new file called `deploy_jobs.py` (or use any name) in the `experiment_management/` folder, and copy the following code in:

```python
import foundations

for _ in range(3):
    foundations.deploy(
        env="scheduler",
        job_directory="experiments/text_generation_simple",
        entrypoint="main.py",
    )
```

Change directory out of the project folder
and run the script we just created

```bash
cd ../..
python experiment_management/deploy_jobs.py
```

Because of the loop we used, this will submit 3 jobs. 

### Explore parameter and architecture space 

At the top of `main.py`, we have a `params` dictionary. This is just a 
configuration 
of parameters for convenience so far. 

Let's refactor a bit so Foundations will track our parameters for us! 

Start by opening the deploy script we created in the `experiment_management/`

Add the following code after `import foundations`

```python
import numpy as np

def get_params():
    params = {
        "rnn_units": np.random.randint(256, 2049),
        "batch_size": np.random.randint(16, 256),
        "embedding_dim": np.random.randint(128, 512),
        "epochs": 30,
        "seq_length": 100,
        "temperature": 0.1,
        "dataset_url": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    }
    return params
```

Then to `foundations.deploy(...)`, add the parameter `params=get_params()`


Now in `main.py` add the following line somewhere after `import foundations`:

```python
params = foundations.load_parameters()
```

Now every time you submit a job, foundations will generate a new set of 
parameters for you which it will show in the GUI and track.

Let's try it out! 

```bash
python experiment_management/deploy_jobs.py
```

This will submit 3 jobs with 3 different sets of parameters.

You can change the number in the loop we created in `deploy_jobs.py` and 
do a larger search (try somewhere between 10 and 20 for now since the trial 
environment includes 10 machines with GPUs). 


## Serving

Foundations provides a standard format for packaging your 
machine learning code so that can be productionized seamlessly.

### Pick a best performing model



### Make code "serveable"

### Ask Foundations to serve it [TODO doesn't exist yet]

### Set up pre-baked web app [TODO doesn't exist yet]


## Load a large model from someone else, use as pretrained model for some new data (GPT)

ALL OF THIS IS STILL TODO

### v1 code, retrieve pre-trained job

### retrain (hopefully)

### serve

