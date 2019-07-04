# Tutorial (TODO title)

Intro TODO

## 1 Run a simple job

We're going to start off with some code we've provided. It's a basic recurrent language model that will train on some text, and learn to generate similar text based on a prompt. 

Note that this code is nothing special right now; just some basic Python libraries and TensorFlow. We don't really have to modify it at all in the beginning to just run it with Foundations. 

### 1.0 Go to the online IDE

[http://35.231.61.216:8443/](http://35.231.61.216:8443/)

### 1.1 Click to run

In the 'Debug' menu, click 'Start Without Debugging' [TODO can this be better]

This is set up to run the entry point, `driver.py`. It can be customized (e.g. your entry point does not *need* to be named "driver"), and it's really just running a simple Foundations command, which looks something like

```
$ foundations deploy --project_name=my_project src/driver.py
```

And that's it! You can also easily deploy Foundations jobs from within Python code using the Foundations library

### 1.2 See latest logs

[TODO there should be a friendlier way, otherwise we'll show it later]

Copy the job_id from the job you just submitted.

Now in the terminal type

```
$ foundations retrieve logs --job_id=<copied_job_id>

```

### Look at GUI

In a new tab, open [https://35.231.61.216:6443/](https://35.231.61.216:6443/). Note that we haven't tracked any metrics or submitted jobs with lots of different parameters yet, but later we'll be able to easily track them here. 

### Receive Slack message

In `utils.py`, [TODO how does a demo user get their own Slack ID]

You can use `post_slack_channel()` to freely post 

## Experiment queueing management

Tracking experiments is powerful if you do a large. We can track all the hyperparameters we try, and all metrics you can calculate in code about any particular experiment.

### Log a metric 

In `driver.py` we have a couple of lines that collect useful information about our model. In this case, we have `loss`, but in other modeling projects you might want to use metrics like accuracy, or AUCROC, or custom business metrics. 

For now, let's track the train loss and test loss metrics we've already calculated. 

Start by adding an import statement at the top of `driver.py`:

```
import foundations
```

Look for the following lines


```
train_loss = model.test(dataset_train, steps_per_epoch_train)
print(train_loss)

test_loss = model.test(dataset_test, steps_per_epoch_test)
print(test_loss)

generated_text = model.generate_text(start_string=u"ROMEO: ", checkpoint_dir='./training_checkpoints', temperature=params['temperature'])
print(generated_text)
 ```
    
 All you need to do to track a metric using foundations is to call `log_metric` on any number you want foundations to save
 
 ```angular2
foundations.log_metric("train loss", train_loss)
foundations.log_metric("test loss", test_loss)
foundations.log_metric("sample output", generated_text[:100])
```

### Add parameters

[TODO, use Foundations hyperparams]
Presently hyperparams already exist, let's store them
[TODO, foundations generate_random_parameters IS COMING SOON]

### Run a job and look at the GUI again


### Run a hyperparameter search [TODO might not work yet] [Distributed workload!]



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


1. Look at our very simple implimentation of a text completion model implemented in TensorFlow `text_generation_simple` from github 
