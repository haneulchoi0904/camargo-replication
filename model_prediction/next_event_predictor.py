# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:35:53 2020

@author: Manuel Camargo
"""
import numpy as np

import utils.support as sup


class NextEventPredictor():

    def __init__(self):
        """constructor"""
        self.model = None
        self.spl = dict()
        self.imp = 'arg_max'

    def predict(self, params, model, spl, imp, vectorizer):
        if isinstance(model, str):
            # model_predictor passes the path of the .h5 file, not a loaded model
            from tensorflow.keras.models import load_model
            model = load_model(model, compile=False)
        self.model = model
        self.spl = spl
        self.imp = imp
        predictor = self._get_predictor(params['model_type'])
        sup.print_performed_task('Predicting next events')
        return predictor(params, vectorizer)

    def _get_predictor(self, model_type):
        # OJO: This is an extension point just incase
        # a different predictor being neccesary
        return self._predict_next_event_shared_cat

    def _predict_next_event_shared_cat(self, parameters, vectorizer):
        """Generate business process suffixes using a keras trained model.
        Args:
            model (keras model): keras trained model.
            prefixes (list): list of prefixes.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            imp (str): method of next event selection.
        """
        # Generation of predictions (all samples at once: the original called model.predict once per
        # sample, which takes hours on the larger logs; the predictions are identical)
        results = list()
        time_dim = parameters['dim']['time_dim']
        n_samples = len(self.spl['prefixes']['activities'])

        def left_pad(seq, dim, attrs=None):
            seq = np.array(seq)
            if attrs is None:
                return np.append(np.zeros(dim), seq, axis=0)[-dim:]
            return np.append(np.zeros((dim, attrs)), seq, axis=0)[-dim:].reshape((dim, attrs))

        x_ac = np.array([left_pad(p, time_dim) for p in self.spl['prefixes']['activities']])
        x_rl = np.array([left_pad(p, time_dim) for p in self.spl['prefixes']['roles']])
        t_attr = self.spl['prefixes']['times'][0].shape[1]
        x_t = np.array([left_pad(p, time_dim, t_attr) for p in self.spl['prefixes']['times']])
        if vectorizer in ['basic']:
            inputs = [x_ac, x_rl, x_t]
        elif vectorizer in ['inter']:
            i_attr = self.spl['prefixes']['inter_attr'][0].shape[1]
            x_in = np.array([left_pad(p, time_dim, i_attr) for p in self.spl['prefixes']['inter_attr']])
            inputs = [x_ac, x_rl, x_t, x_in]
        all_preds = self.model.predict(inputs, batch_size=1024, verbose=0)
        for i in range(n_samples):
            preds = [all_preds[0][i:i + 1], all_preds[1][i:i + 1], all_preds[2][i:i + 1]]
            if self.imp == 'random_choice':
                pos = np.random.choice(np.arange(0, len(preds[0][0])), p=preds[0][0])
                pos1 = np.random.choice(np.arange(0, len(preds[1][0])), p=preds[1][0])
            elif self.imp == 'arg_max':
                pos = np.argmax(preds[0][0])
                pos1 = np.argmax(preds[1][0])
            predictions = [pos, pos1, preds[2][0][0]]
            if not parameters['one_timestamp']:
                predictions.extend([preds[2][0][1]])
            results.append(
                self.create_result_record(i, self.spl, predictions, parameters))
        sup.print_done_task()
        return results

    def create_result_record(self, index, spl, preds, parms):
        record = dict()
        record['ac_prefix'] = spl['prefixes']['activities'][index]
        record['ac_expect'] = spl['next_evt']['activities'][index]
        record['ac_pred'] = preds[0]
        record['rl_prefix'] = spl['prefixes']['roles'][index]
        record['rl_expect'] = spl['next_evt']['roles'][index]
        record['rl_pred'] = preds[1]
        if parms['one_timestamp']:
            record['tm_prefix'] = [self.rescale(
                x, parms, parms['scale_args']) 
                for x in spl['prefixes']['times'][index]]
            record['tm_expect'] = self.rescale(
                spl['next_evt']['times'][index][0],
                parms, parms['scale_args'])
            record['tm_pred'] = self.rescale(
                preds[2], parms, parms['scale_args'])
        else:
            # Duration
            record['dur_prefix'] = [self.rescale(
                x[0], parms, parms['scale_args']['dur'])
                for x in spl['prefixes']['times'][index]]
            record['dur_expect'] = self.rescale(
                spl['next_evt']['times'][index][0], parms,
                parms['scale_args']['dur'])
            record['dur_pred'] = self.rescale(
                preds[2], parms, parms['scale_args']['dur'])
            # Waiting
            record['wait_prefix'] = [self.rescale(
                x[1], parms, parms['scale_args']['wait'])
                for x in spl['prefixes']['times'][index]]
            record['wait_expect'] = self.rescale(
                spl['next_evt']['times'][index][1], parms,
                parms['scale_args']['wait'])
            record['wait_pred'] = self.rescale(
                preds[3], parms, parms['scale_args']['wait'])
        return record

    @staticmethod
    def rescale(value, parms, scale_args):
        if parms['norm_method'] == 'lognorm':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            value = (value * (max_value - min_value)) + min_value
            value = np.expm1(value)
        elif parms['norm_method'] == 'normal':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            value = (value * (max_value - min_value)) + min_value
        elif parms['norm_method'] == 'standard':
            mean = scale_args['mean']
            std = scale_args['std']
            value = (value * std) + mean
        elif parms['norm_method'] == 'max':
            max_value = scale_args['max_value']
            value = np.rint(value * max_value)
        elif parms['norm_method'] is None:
            value = value
        else:
            raise ValueError(parms['norm_method'])
        return value
