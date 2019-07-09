"""
This file is used for your model code, and for organizing and executing Foundations stages.

Below we import foundations, as well as a function from our model file that we'll use as a stage.

We then tell Foundations how to run by defining a path to a configuration file.

Next we define a project name using `foundations.set_project_name()`.

We define the experiment to run as `result` and then use Foundations's `.run()` method which tells Foundations to run these stages.

Then you can run the driver file with `python main.py` to send the experiment off to be run. To check results, see the `/results` directory where you'll read and interact with results.
"""
from utils import post_slack_channel
import time

try:
    from preprocessing import download_data, preprocess_data
    from model import Model
    from utils import get_params, save_preprocessors, load_preprocessors

    path_to_file = download_data('https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt', 'shakespeare.txt')

    dataset_train, dataset_test, steps_per_epoch_train, steps_per_epoch_test, vocab, char2idx, idx2char = preprocess_data(path_to_file, params)

    # Save preprocessors for serving
    save_preprocessors(char2idx, idx2char, vocab)

    model = Model(vocab,
                  embedding_dim=params['embedding_dim'],
                  rnn_units=params['rnn_units'],
                  batch_size=params['batch_size'],
                  char2idx=char2idx,
                  idx2char=idx2char)

    model.train(dataset_train,
                steps_per_epoch=steps_per_epoch_train,
                checkpoint_dir='./training_checkpoints',
                epochs=params['epochs'],)

    train_loss = model.test(dataset_train, steps_per_epoch_train)
    print(train_loss)

    test_loss = model.test(dataset_test, steps_per_epoch_test)
    print(test_loss)

    model.set_test_mode(checkpoint_dir='./training_checkpoints')

    start_time = time.time()
    generated_text = model.generate(start_string=u"ROMEO: ", temperature=params['temperature'])
    print('synthesis time: {}'.format(time.time() - start_time))
    print(generated_text)
    
except Exception as e:
    import traceback
    
    trace = traceback.format_exc()
    post_slack_channel('Job failed. Catching Exception:\n{}'.format(trace))
    raise e
