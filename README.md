# Python Environment

```bash
conda create -n flair_env python=3.11
python -m pip install -r requirements.txt
```

**NOTE:** refer to the official [PyTorch documentation](https://pytorch.org/get-started/locally/) to install it locally.


# Main
Running the following command will:
* download the data if needed
* train the model
* evaluate the model at each epoch on the validation set
* evaluate the model on the test set at the end of the training
* save the checkpoint of the model, metrics for train, validation and test set, alongside a copy of the config `yaml` file

```bash
python -m src.main [CONFIG.yaml]
```

# Plot metrics
The metrics, per epoch, are saved for the train and validation. One could plot them by the following command:

```bash
python -m src.plots.visualize --save --json-file [FILE.json]
```

**NOTE:** to plot the metrics for all the `json` one could use [GNU Parallel](https://www.gnu.org/software/parallel/):

```bash
ls [FOLDER/**/*.json] | parallel python -m src.plots.visualize --save --json-filea {}
```
Or to aggregate the metrics of the train, validation and test sets: 
```bash
python -m src.plots.aggregate --save --directory [LOGS_DIR]
```
