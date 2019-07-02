"""
This file is used for your model code, and for organizing and executing Foundations stages.

Below we import foundations, as well as a function from our model file that we'll use as a stage.

We then tell Foundations how to run by defining a path to a configuration file.

Next we define a project name using `foundations.set_project_name()`.

We define the experiment to run as `result` and then use Foundations's `.run()` method which tells Foundations to run these stages.

Then you can run the driver file with `python driver.py` to send the experiment off to be run. To check results, see the `/results` directory where you'll read and interact with results.
"""
from utils import post_slack_channel
job_id = -1

try:
    import foundations
    from preprocessing import download_data, preprocess_data
    from model import Model
    from utils import get_params
    
    path_to_file = download_data('https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt', 'shakespeare.txt')
    
    params = get_params()
    
    dataset, steps_per_epoch, vocab, char2idx, idx2char = preprocess_data(path_to_file, params)
    
    model = Model(vocab,
                  embedding_dim=params['embedding_dim'],
                  rnn_units=params['rnn_units'],
                  batch_size=params['batch_size'],
                  char2idx=char2idx,
                  idx2char=idx2char)
    
    model.train(dataset,
                steps_per_epoch=steps_per_epoch,
                checkpoint_dir='./training_checkpoints',
                epochs=params['epochs'],
                job_id=job_id)
    
    train_loss = model.test(dataset, steps_per_epoch)
    foundations.log_metric("train_loss", train_loss)
    print(train_loss)
    
    generated_text = model.generate_text(start_string=u"ROMEO: ", checkpoint_dir='./training_checkpoints', temperature=params['temperature'])
    
    print(generated_text)
    
except Exception as e:
    post_slack_channel('Job failed. Catching Exception:\n{}'.format(e),
                       job_id=job_id)

