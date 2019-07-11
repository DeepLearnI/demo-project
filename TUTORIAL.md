
<img style="float: left;" src="https://dessa.com/wp-content/uploads/2018/05/dessa_logo.svg">
<p align="right"> Powered by <img src="https://storage.googleapis.com/gweb-cloudblog-publish/images/f4xvje.max-200x200.PNG" height="50" width="50" align="right">
</p>
<br>
<hr>


# Welcome to Foundations

*Estimated time: 20 minutes*

Welcome to the Foundations trial environment! 

In this tutorial we'll go through the process of optimizing and 
serving a simple text generator
model (similar to GPT-2) using Foundations.

This trial environment provides you with 
a fully managed Foundations setup including:


* 10 GPUs
* Foundations, TensorFlow, and the Python scientific stack 
(NumPy, pandas, etc) pre-installed 
* A fully-featured in-browser IDE


_Keep in mind Foundations is infrastructure-agnostic and can be set up on premise 
or on cloud depending on your needs._



In this trial we will start by taking some basic model code and using it 
to explore some of Foundations' most important features.


This is what we'll achieve today:

1. Without adding any Foundations-specific code at all, 
we will submit code to be
 executed on remote machines with multiple GPUs.

1. We will then add our first lines of Foundations code to track and 
share metrics of our choosing.

1. We will optimise our model using Foundations' parameter/architecture
search 
feature and see how Foundations can execute multiple jobs 
using all available computation resources. 

1. We will then see how Foundations automatically tracks the parameters and 
results of all of these in an experiment dashboard, enabling full reproducibility.

1. Finally, we'll select the best model and serve it to a demo web app.  


## 1 Hello, world!

To the right of this pane, you will see `main.py`. This is a piece of code that was 
quickly assembled by one of our machine learning engineers _without using Foundations_. 

The model is a language generator. We will train it on 
some Shakespearean text, and the resultant model 
will be able to synthesize new text that sounds 
(ostensibly) like Shakespeare. 
 
We're going to optimize the model performance using an architecture and parameter
 search. 


### 1.1 Architecture and parameter search


#### 1.1.1 Add metrics

In `main.py` we already have a couple of lines that print 
useful information about our model. It's easy to get Foundations to log them. 

For now, let's track the train loss and test loss metrics we already have. [TODO rewrite]

Start by adding an import statement to the top of `main.py`:

```
import foundations
```

Now look at line <PLACEHOLDER>


```python
train_loss = model.test(dataset_train, steps_per_epoch_train)
print(train_loss)

test_loss = model.test(dataset_test, steps_per_epoch_test)
print(test_loss)

generated_text = model.generate_text(start_string=u"ROMEO: ", checkpoint_dir='./training_checkpoints', temperature=params['temperature'])
print(generated_text)
 ```
    
To track any performance metric using Foundations you can 
call `log_metric` on any number or string:
 
 ```python
foundations.log_metric("train loss", train_loss)
foundations.log_metric("test loss", test_loss)
foundations.log_metric("sample output", generated_text)
```


#### 1.1.2 Create a job deployment script

 Without Foundations, running a search over many 
 architectures and sets of hyperparameters
 is messy and difficult to manage.  Foundations makes this
  straightforward! We're going to 
 write a simple script to immediately kick off 
 100 jobs on our cluster.

Create a new file called `submit_jobs.py` 
in the `experiment_management/` folder, and add in the 
following code:


```python
import foundations
import numpy as np

NUM_JOBS = 100

# Get params returns randomly generated architecture specifications 
# and hyperparameters in the form of a dictionary
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
    
# A loop that calls the deploy method from the  
# Foundations SDK which takes in a parameters dictionary
# and the entrypoint script for our code
for _ in range(NUM_JOBS):
    foundations.deploy(
        env="scheduler",
        job_directory="experiments/text_generation_simple",
        entrypoint="main.py",
        project_name="a_project_name"
        params=get_params(),
    )
```

At the bottom of this window you'll see a terminal. 
Type the following command
 to launch the script we just wrote:
 

```bash
$ python experiment_management/submit_jobs.py
```

That's it! We're now using the full capacity 
of our compute resources to explore our architecture and 
parameter space through 100 models training 
concurrently.

Let's take a look at how they're doing.


### 1.2 Dashboard

Foundations provides a Dashboard that allows teams to monitor 
and manage 
multiple projects across a cluster. We can take a look at the 
parameters and performance metrics of the 100 jobs we 
submitted. 


Click [here](DASHBOARD_URL) to open the Dashboard.


[TODO place dot legend]

Some jobs will already be completed. We added a sample
of generated output as a metric — hover 
over a few examples 
to check how our initial models are doing.

---
---
---



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

