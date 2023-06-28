import numpy as np
import awkward as ak
import tqdm
import time
import torch

from collections import defaultdict, Counter
from weaver.utils.nn.metrics import evaluate_metrics
from weaver.utils.data.tools import _concat
from weaver.utils.logger import _logger

def get_layer_output(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                        eval_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                                      'mean_gamma_deviance'],
                        tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    actual_output = []
    layer_output = []
    input_features = []
    vec_features = []
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].float()
                num_examples = label.shape[0]
                label = label.to(dev)
                model_output = model(*inputs)

                actual_output.append(model_output[0].cpu().numpy())
                layer_output.append(model_output[1].cpu().numpy())
                input_features.append(model_output[2].cpu().numpy())
                vec_features.append(model_output[3].cpu().numpy())

                #print(model_output[1].size())
                
                #preds = model_output[0].squeeze().float()

                #scores.append(preds.detach().cpu().numpy())
                #for k, v in y.items():
                #    labels[k].append(v.cpu().numpy())
                    
                #if not for_training:
                #    for k, v in Z.items():
                #        observers[k].append(v.cpu().numpy())

                #loss = 0 if loss_func is None else loss_func(preds, label).item()

                #num_batches += 1
                #count += num_examples
                #total_loss += loss * num_examples
                #e = preds - label
                #abs_err = e.abs().sum().item()
                #sum_abs_err += abs_err
                #sqr_err = e.square().sum().item()
                #sum_sqr_err += sqr_err

                #tq.set_postfix({
                #    'Loss': '%.5f' % loss,
                #    'AvgLoss': '%.5f' % (total_loss / count),
                #    'MSE': '%.5f' % (sqr_err / num_examples),
                #    'AvgMSE': '%.5f' % (sum_sqr_err / count),
                #    'MAE': '%.5f' % (abs_err / num_examples),
                #    'AvgMAE': '%.5f' % (sum_abs_err / count),
                #})

                #if tb_helper:
                #    if tb_helper.custom_fn:
                #        with torch.no_grad():
                #            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                #                                mode='eval' if for_training else 'test')

                #if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                #    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))

    if tb_helper:
        #tb_mode = 'eval' if for_training else 'test'
        #tb_helper.write_scalars([
        #    ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
        #    ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
        #    ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
        #    ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)


    actual_output = np.concatenate(actual_output,axis=0)
    layer_output = np.concatenate(layer_output,axis=0)
    input_features = np.concatenate(input_features,axis=0)
    vec_features = np.concatenate(vec_features,axis=0)
    print(model_output[0].size())
    print(model_output[1].size())
    print(model_output[2].size())
    #scores = np.concatenate(scores)
    #labels = {k: _concat(v) for k, v in labels.items()}
    #metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    #_logger.info('Evaluation metrics: \n%s', '\n'.join(
    #    ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_loss / count
    else:
        # convert 2D labels/scores
        #observers = {k: _concat(v) for k, v in observers.items()}
        return actual_output, layer_output, input_features, vec_features






def evaluate_regression(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                        eval_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                                      'mean_gamma_deviance'],
                        tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    counter = 0
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            print("-----------")
            print("counter: ", counter)
            counter += 1
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].float()
                num_examples = label.shape[0]
                label = label.to(dev)
                model_output = model(*inputs)
                preds = model_output.squeeze().float()

                scores.append(preds.detach().cpu().numpy())
                for k, v in y.items():
                    labels[k].append(v.cpu().numpy())
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.cpu().numpy())

                loss = 0 if loss_func is None else loss_func(preds, label).item()

                num_batches += 1
                count += num_examples
                total_loss += loss * num_examples
                e = preds - label
                abs_err = e.abs().sum().item()
                sum_abs_err += abs_err
                sqr_err = e.square().sum().item()
                sum_sqr_err += sqr_err

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'MSE': '%.5f' % (sqr_err / num_examples),
                    'AvgMSE': '%.5f' % (sum_sqr_err / count),
                    'MAE': '%.5f' % (abs_err / num_examples),
                    'AvgMAE': '%.5f' % (sum_abs_err / count),
                })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
            ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores = np.concatenate(scores)
    print("-----------")
    print("-----------")
    print("scores: ", scores.shape)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_loss / count
    else:
        # convert 2D labels/scores
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_loss / count, scores, labels, observers
