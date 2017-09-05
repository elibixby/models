from __future__ import division

import six

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import device_setter


class VariableStrategy:
  CPU = 'cpu'
  GPU = 'gpu'


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
  if ps_ops is None:
    ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

  if ps_strategy is None:
    ps_strategy = device_setter._RoundRobinStrategy(num_devices)
  if not six.callable(ps_strategy):
    raise TypeError("ps_strategy must be callable")

  def _local_device_chooser(op):
    current_device = pydev.DeviceSpec.from_string(op.device or "")

    node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
    if node_def.op in ps_ops:
      ps_device_spec = pydev.DeviceSpec.from_string(
          '/{}:{}'.format(ps_device_type, ps_strategy(op)))

      ps_device_spec.merge_from(current_device)
      return ps_device_spec.to_string()
    else:
      worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
      worker_device_spec.merge_from(current_device)
      return worker_device_spec.to_string()
  return _local_device_chooser


def _split_batch(features, labels, num_shards):
  with tf.name_scope('split_inputs'):
    with tf.device('/cpu:0'):
      if isinstance(features, dict):
        feature_shards = [{} for _ in range(num_shards)]
        for name, tensor in six.iteritems(features):
          for i, shard in enumerate(tf.split(tensor, num_shards)):
            feature_shards[i][name] = shard
      else:
        feature_shards = tf.split(features, num_shards)
    if labels is None:
      label_shards = None
    else:
      label_shards = tf.split(labels, num_shards)
  return feature_shards, label_shards


def _dict_concat(*dicts):
  list_dict = {}
  for d in dicts:
    for k, v in six.iteritems(d):
      list_dict.setdefault(k, []).append(v)
  return list_dict


def _avg_tensor_dicts(*tensor_dicts):
  return {
      name: tf.reduce_mean(tf.stack(tensors), axis=0)
      if len(tensors) > 1 else tensors[0]
      for name, tensors in six.iteritems(_dict_concat(*tensor_dicts))
  }


def _concat_tensor_dicts(*tensor_dicts):
  return {
      name: tf.concat(tensors, axis=0) if len(tensors) > 1 else tensors[0]
      for name, tensors in six.iteritems(_dict_concat(*tensor_dicts))
  }


def get_available_devices(device_type):
  local_device_protos = device_lib.list_local_devices()
  return [device.name
          for device in local_device_protos
          if device.device_type == device_type]


class TowerEstimator(tf.estimator.Estimator):

  def __init__(self,
               model_fn,
               optimizer_fn,
               train_device_type='GPU',
               eval_device_type='GPU',
               predict_devices=None,
               variable_strategy=VariableStrategy.CPU,
               sync_device='/cpu:0',
               sync_replicas=True,
               update_from_all_towers=False,
               *args,
               **kwargs):

    self._devices = {
        ModeKeys.TRAIN: get_available_devices(train_device_type),
        ModeKeys.EVAL: get_available_devices(eval_device_type),
        ModeKeys.PREDICT: predict_devices or ['/cpu:0']
    }
    self._sync_device = sync_device
    self._optimizer_fn = optimizer_fn
    self._sync_replicas = sync_replicas
    self._tower_model_fn = model_fn
    self._update_from_all_towers = update_from_all_towers
    self._ps_device_type = variable_strategy

    super(TowerEstimator, self).__init__(model_fn=self._wrapped_model_fn, *args, **kwargs)

  def _get_towers(self, mode, features, labels, params, devices):
    tower_specs = []
    for i, device in enumerate(devices):
      device_setter = local_device_setter(
          num_devices=len(devices),
          worker_device=device,
          ps_device_type=self._ps_device_type)
      with tf.variable_scope('tower_vars', reuse=bool(i != 0)):
        with tf.name_scope('tower_{}'.format(i)):
          with tf.device(device_setter):
            tower_spec = self._tower_model_fn(
                mode=mode, features=features[i], labels=labels[i], params=params)
            tower_specs.append(tower_spec)
    return tower_specs

  def _wrapped_model_fn(self, mode, features, labels, params):
    if len(self._devices[mode]) > 1:
      feature_shards, label_shards = _split_batch(
          features, labels, len(self._devices[mode]))
    else:
      feature_shards, label_shards = ([features], [labels])
    tower_specs = self._get_towers(
        mode,
        feature_shards,
        label_shards,
        params,
        self._devices[mode]
    )
    if mode == ModeKeys.TRAIN:
      return self._train_spec(tower_specs, params=params)
    if mode == ModeKeys.EVAL:
      return self._eval_spec(tower_specs)
    if mode == ModeKeys.PREDICT:
      return self._predict_spec(tower_specs)

  def _get_average_loss(self, tower_specs):
    if len(tower_specs) > 1:
      with tf.device(self._sync_device):
        return tf.multiply(
            tf.add_n([tower_spec.loss for tower_spec in tower_specs]),
            1 / len(tower_specs),
            name='loss'
        )
    else:
      return tf.identity(tower_specs[0].loss, name='loss')

  def _predict_spec(self, tower_specs):
    old_estimator_spec = tower_specs[0]._asdict()

    with tf.device(self._sync_device):
      old_estimator_spec['predictions'] = _concat_tensor_dicts(
          *[tower_spec.predictions for tower_spec in tower_specs])

      export_output_dict = _dict_concat(
          *[tower_spec.export_outputs for tower_spec in tower_specs])

      export_outputs = {}
      for name, export_output_list in six.iteritems(export_output_dict):
        if isinstance(export_output_list[0], tf.estimator.export.PredictOutput):
          export_outputs[name] = tf.estimator.export.PredictOutput(
              outputs=_concat_tensor_dicts(
                  *[export_output.outputs for export_output in export_output_list]
              )
          )
        elif isinstance(export_output_list[0], tf.estimator.export.RegressionOutput):
          export_output[name] = tf.estimator.export.RegressionOutput(
              value=tf.concat(
                  [export_output.value for export_output in export_output_list],
                  axis=0
              ) if len(export_output_list) > 1 else export_output_list[0].value
          )
        elif isinstance(export_output_list[0], tf.estimator.export.ClassificationOutput):
          if export_output_list[0].scores is not None:
              scores = tf.concat(
                  [export_output.scores for export_output in export_output_list],
                  axis=0
              ) if len(export_output_list) > 1 else export_output_list[0].scores
          else:
              scores = None
          if export_output_list[0].classes is not None:
              classes = tf.concat(
                  [export_output.classes for export_output in export_output_list],
                  axis=0
              ) if len(export_output_list) > 1 else export_output_list[0].classes
          else:
              scores = None

          export_output[name] = tf.estimator.export.ClassificationOutput(
              scores=scores, classes=classes)
    old_estimator_spec['export_outputs'] = export_outputs
    return tf.estimator.EstimatorSpec(ModeKeys.PREDICT, **old_estimator_spec)


  def _eval_spec(self, tower_specs):
    old_estimator_spec = tower_specs[0]._asdict()
    old_estimator_spec['loss'] = self._get_average_loss(tower_specs)
    eval_metric_ops_lists = {}
    for tower_spec in tower_specs:
      metrics = tower_spec.eval_metric_ops or {}
      for name, (metric_tensor, update_op) in six.iteritems(metrics):
        metric_lists = eval_metric_ops_lists.setdefault(name, ([], []))
        metric_lists[0].append(metric_tensor)
        metric_lists[1].append(update_op)

    with tf.device(self._sync_device):
      eval_metric_ops = {}
      for name, (metric_tensors, update_ops) in six.iteritems(
          eval_metric_ops_lists):
        eval_metric_ops[name] = (
            tf.reduce_mean(tf.stack(metric_tensors), axis=0),
            tf.group(*update_ops)
        )

    old_estimator_spec['eval_metric_ops'] = eval_metric_ops
    return tf.estimator.EstimatorSpec(ModeKeys.EVAL, **old_estimator_spec)

  def _train_spec(self, tower_specs, params):
    with tf.name_scope('tower_0') as name_scope:
      update_ops = tf.get_collection(
          tf.GraphKeys.UPDATE_OPS,
          None if self._update_from_all_towers else name_scope
      )

    grad_lists = {}
    for tower_spec in tower_specs:
      with tf.device(tower_spec.loss.device):
        variables = tf.trainable_variables()
        gradients = tf.gradients(tower_spec.loss, variables)
        for var, grad in zip(variables, gradients):
          if grad is not None:
            grad_lists.setdefault(var, []).append(grad)

    averaged_grads = []
    with tf.name_scope('gradient_averaging'):
      for var, grads in six.iteritems(grad_lists):
        with tf.device(var.device):
          if len(grads) == 1:
            avg_grad = grads[0]
          else:
            avg_grad = tf.multiply(tf.add_n(grads), 1 / len(grads))
        averaged_grads.append((avg_grad, var))


    old_estimator_spec = tower_specs[0]._asdict()

    old_estimator_spec['loss'] = self._get_average_loss(tower_specs)
    with tf.device(self._sync_device):
      if self._sync_replicas and self.config.num_worker_replicas:
        # Master is not local
        if self.config.master:
           num_workers = self.config.num_worker_replicas + 1
        else:
           num_workers = self.config.num_worker_replicas
        optimizer = tf.train.SyncReplicasOptimizer(
            self._optimizer_fn(params),
            replicas_to_aggregate=num_workers
        )
        tf.logging.info(self.config)
        sync_replicas_hook = optimizer.make_session_run_hook(
            self.config.is_chief)
        old_hooks = old_estimator_spec.get('training_hooks', ())
        old_estimator_spec['training_hooks'] = list(old_hooks).append(
            sync_replicas_hook)
      else:
        optimizer = self._optimizer_fn(params)

      with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(
            averaged_grads,
            global_step=tf.train.get_global_step())

        if self._sync_replicas and self.config.num_worker_replicas:
          optimizer.ready_for_local_init_op.mark_used()

        old_estimator_spec['train_op'] = train_op
    print(old_estimator_spec)
    return tf.estimator.EstimatorSpec(ModeKeys.TRAIN, **old_estimator_spec)
