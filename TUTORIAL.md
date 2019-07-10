![Dessa Logo](https://dessa.com/wp-content/uploads/2018/05/dessa_logo.svg)

# Welcome to Foundations

*Estimated time: 20 minutes*

Welcome to the Foundations trial environment! 

In this tutorial we'll go through the process of optimizing and serving a simple text generator
model (similar to GPT-2) using Foundations.

This trial environment provides you with 
a fully managed Foundations setup including:

* 5 GPUs
* Foundations, TensorFlow, and the Python scientific stack 
(NumPy, pandas, etc) pre-installed 
* A fully-featured in-browser IDE


_Keep in mind Foundations is infrastructure-agnostic and can be set up on premise 
or on cloud depending on your needs._



In this trial we will start by some basic model code and using it 
to explore some of Foundations' most important features.


This is what we'll achieve today:

1. Without any Foundations-specific code at all, we will submit our code to be
 executed on remote machines with multiple GPUs.

1. We will then add our first lines of Foundations code to track and 
share metrics of our choosing.

1. We will optimise our model using Foundations' parameter and architecture
search 
feature and see how Foundations can execute multiple jobs 
using all available computation resources. 

1. We will then see how Foundations automatically tracks the parameters and 
results of all of these in an experiment log, enabling full reproducibility.

1. Finally, we'll select the best model and serve it to a demo web app.  



## 1 Hello, world!

To the right of this pane, you will see `main.py`. This is a piece of code that was 
quickly written by one of our machine learning engineers _without using Foundations_. 

The model is a language generator. We train it on 
some Shakespearean text, and the resultant model 
 will be able to synthesize new text that sounds (ostensibly) like Shakespeare. 
 
Note that this code is nothing special right now; just some basic Python libraries and 
TensorFlow. We don't really have to modify it at all in the beginning to just submit it 
to a cluster of GPU machines using Foundations. 

The code is set to use a very short training time, so the early output may be 
low quality or even nonsensical; this is just for speed, we'll train the model for longer 
later
in the tutorial.


### 1.1 Submit job

Using one command, we're going to take our code and run it on a remote 
server with a GPU.

Underneath you'll see a terminal. 

Use the command below to `cd` into the project directory:

```bash
$ cd experiments/text_generation_simple
```

Go ahead and copy this command into the terminal

```bash
$ foundations deploy --env scheduler 
```

Submitting a job like this will stream the output live; 
you can stop the streaming at any time using Ctrl+C, 
the job will continue running.

Let's break down this command briefly: 

* `foundations deploy` submits a job
* `--env` looks for a configuration file by that name. In a long-term 
project you might have need to submit jobs to different environments, 
or have a configuration file for local submissions. 

We are making use of defaults for a few others:

* `--project-name` is a user-specified project title. It defaults to the 
directory name. You can freely create new ones whenever you need
* `--entrypoint` is the name of the script to run, it defaults 
to `main.py` but can be specified manually

For more guidance on this command, you can always use

```bash
foundations deploy --help
```


### 1.2 Look at GUI

In a new tab, open [https://<IP>:<PORT>/projects](https://<IP>:<PORT>/projects). 
Click on your Project.

Note that we haven't tracked any metrics or submitted jobs with lots of 
different parameters yet, but later in this tutorial we'll easily track them here. 

The little icons indicate the status of a job. Green means success, yellow means currently running,
red indicates a failure or unclean exit, and grey means the job is queued and not yet running.


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

Experiment management is a powerful feature. We can track all the 
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

Run these commands below to change directory out of the 
project folder
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


Now in `main.py` add the following line somewhere after line 15:

```python
params = foundations.load_parameters()
```

Now every time you submit a job, foundations will generate a new set of 
parameters for you which it will show in the GUI and track.

Let's try it out by deploying a bunch of jobs 
using just the following command in terminal: 

```bash
python experiment_management/deploy_jobs.py
```

This will submit 3 jobs with 3 different sets of parameters.


## Serving

Foundations provides a standard format for packaging your 
machine learning code so that can be productionized seamlessly.


### Make code "serveable"

Create a new file called `predict.py` file.

```python
from utils import load_preprocessors
from model import Model
import foundations


char2idx, idx2char, vocab = load_preprocessors()

params = foundations.load_parameters()

model = Model(vocab,
              embedding_dim=params['embedding_dim'],
              rnn_units=params['rnn_units'],
              batch_size=1,
              char2idx=char2idx,
              idx2char=idx2char)

model.load_saved_model(checkpoint_dir='./training_checkpoints')

def generate_prediction(input_text):
    generated_text = model.generate(start_string=input_text, temperature=params['temperature'])
    return generated_text
```

This will just load a saved model, and it surfaces a function called `predict`.

Now we just need a configuration file. Create a new file called 
`foundations_package_manifest.yaml` and paste the following text into it:

```yaml
entrypoints:
    predict:
        module: predict
        function: generate_prediction

```

Once a manifest has been added to your root directory, 
you will need to launch a new Foundations job in order for 
Foundations to be able to directly serve it.


### Do a hyperparameter search to get a good model!


Change the number in the loop we created in `deploy_jobs.py` to do 
a larger search, perhaps between 10 and 20 jobs since our
environment includes 10 machines with GPUs. Set the `epochs` to somewhere around 30

Now check the GUI and notice the different parameters we now have 
for each job. As the jobs finish, you'll be able to compare
the performance for the different sets of parameters we've 
tested. 

### Select the best job

Look for the job with the lowest `test_loss` or perhaps your favourite generated
text example! Copy that `job_id`

### Ask Foundations to serve it 

In the terminal, enter

```bash
foundations serve start <JOB_ID>
```

Foundations automatically retrieves the bundle associated with the job_id 
and wraps it in a REST API server, where requests can be made to the entrypoints, 
specified by the `foundations_package_manifest.yaml` file

### Set up pre-baked web app [TODO doesn't exist yet]


Go to the provided WebApp URL. For the Model Name, please use the IP address 
given in the Slack message. Now try getting generated text
from your served model!


## Load a large model from someone else, use as pretrained model for some new data (GPT)


### v1 code, retrieve pre-trained job

### retrain (hopefully)

### serve

