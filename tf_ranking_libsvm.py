# Copyright 2020 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""TF Ranking sample code for LETOR datasets in LibSVM format.
WARNING: All data sets are loaded into memory in this sample code. It is
for small data sets whose sizes are < 10G.
A note on the LibSVM format:
--------------------------------------------------------------------------
Due to the sparse nature of features utilized in most academic datasets for
learning to rank such as LETOR datasets, data points are represented in the
LibSVM format. In this setting, every line encapsulates features and a (graded)
relevance judgment of a query-document pair. The following illustrates the
general structure:
<relevance int> qid:<query_id int> [<feature_id int>:<feature_value float>]
For example:
1 qid:10 32:0.14 48:0.97  51:0.45
0 qid:10 1:0.15  31:0.75  32:0.24  49:0.6
2 qid:10 1:0.71  2:0.36   31:0.58  51:0.12
0 qid:20 4:0.79  31:0.01  33:0.05  35:0.27
3 qid:20 1:0.42  28:0.79  35:0.30  42:0.76
In the above example, the dataset contains two queries. Query "10" has 3
documents, two of which relevant with grades 1 and 2. Similarly, query "20"
has 1 relevant document. Note that query-document pairs may have different
sets of zero-valued features and as such their feature vectors may only
partly overlap or not at all.
--------------------------------------------------------------------------
Sample command lines:
OUTPUT_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train.txt && \
VALI=tensorflow_ranking/examples/data/vali.txt && \
TEST=tensorflow_ranking/examples/data/test.txt && \
rm -rf $OUTPUT_DIR && \
bazel build -c opt \
tensorflow_ranking/examples/tf_ranking_libsvm_py_binary && \
./bazel-bin/tensorflow_ranking/examples/tf_ranking_libsvm_py_binary \
--train_path=$TRAIN \
--vali_path=$VALI \
--test_path=$TEST \
--output_dir=$OUTPUT_DIR \
--num_features=136
You can use TensorBoard to display the training results stored in $OUTPUT_DIR.
Notes:
  * Use --alsologtostderr if the output is not printed into screen.
  * In addition, you can enable multi-objective learning by adding the following
  flags: --secondary_loss=<the secondary loss key>.
"""

from absl import flags

import numpy as np
import six
import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow_ranking.python import utils

flags.DEFINE_string("train_path", None, "Input file path used for training.")
flags.DEFINE_string("vali_path", None, "Input file path used for validation.")
flags.DEFINE_string("test_path", None, "Input file path used for testing.")
flags.DEFINE_string("output_dir", None, "Output directory for models.")
flags.DEFINE_string("query_extraction", 'binary', "Type of relevance for the queries, binary ou continuous.")
flags.DEFINE_string("query_size", 10, "Number of words per query.")

flags.DEFINE_integer("train_batch_size", 32, "The batch size for training.")
flags.DEFINE_integer("num_train_steps", 100000, "Number of steps for training.")

flags.DEFINE_float("learning_rate", 0.01, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.5, "The dropout rate before output layer.")
flags.DEFINE_list("hidden_layer_dims", ["256", "128", "64"],
                  "Sizes for hidden layers.")

flags.DEFINE_integer("num_features", 600, "Number of features per document.")
flags.DEFINE_integer("list_size", 10, "List size used for training.")
flags.DEFINE_integer("group_size", 1, "Group size used in score function.")

flags.DEFINE_string("loss", "pairwise_logistic_loss",
                    "The RankingLossKey for the primary loss function.")
flags.DEFINE_string(
    "secondary_loss", None, "The RankingLossKey for the secondary loss for "
    "multi-objective learning.")
flags.DEFINE_float(
    "secondary_loss_weight", 0.5, "The weight for the secondary loss in "
    "multi-objective learning.")

FLAGS = flags.FLAGS

_PRIMARY_HEAD = "primary_head"
_SECONDARY_HEAD = "secondary_head"


def _use_multi_head():
  """Returns True if using multi-head."""
  return FLAGS.secondary_loss is not None


class IteratorInitializerHook(tf.estimator.SessionRunHook):
  """Hook to initialize data iterator after session is created."""

  def __init__(self):
    super(IteratorInitializerHook, self).__init__()
    self.iterator_initializer_fn = None

  def after_create_session(self, session, coord):
    """Initialize the iterator after the session has been created."""
    del coord
    self.iterator_initializer_fn(session)


def example_feature_columns():
  """Returns the example feature columns."""
  feature_names = ["{}".format(i) for i in range(FLAGS.num_features)]
  return {
      name:
      tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
      for name in feature_names
  }


def load_libsvm_data(path, list_size):
  """Returns features and labels in numpy.array."""

  def _parse_line(line):
    """Parses a single line in LibSVM format."""
    tokens = line.split("#")[0].split()
    assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
    label = float(tokens[0])
    qid = tokens[1]
    kv_pairs = [kv.split(":") for kv in tokens[2:]]
    features = {k: float(v) for (k, v) in kv_pairs}
    return qid, features, label

  tf.compat.v1.logging.info("Loading data from {}".format(path))

  # The 0-based index assigned to a query.
  qid_to_index = {}
  # The number of docs seen so far for a query.
  qid_to_ndoc = {}
  # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
  # a shape of [num_queries, list_size]. We use list for each of them due to the
  # unknown number of quries.
  feature_map = {k: [] for k in example_feature_columns()}
  label_list = []
  total_docs = 0
  discarded_docs = 0
  with open(path, "rt") as f:
    for line in f:
      qid, features, label = _parse_line(line)
      if qid not in qid_to_index:
        # Create index and allocate space for a new query.
        qid_to_index[qid] = len(qid_to_index)
        qid_to_ndoc[qid] = 0
        for k in feature_map:
          feature_map[k].append(np.zeros([list_size, 1], dtype=np.float32))
        label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
      total_docs += 1
      batch_idx = qid_to_index[qid]
      doc_idx = qid_to_ndoc[qid]
      qid_to_ndoc[qid] += 1
      # Keep the first 'list_size' docs only.
      if doc_idx >= list_size:
        discarded_docs += 1
        continue
      for k, v in six.iteritems(features):
        assert k in feature_map, "Key {} not found in features.".format(k)
        feature_map[k][batch_idx][doc_idx, 0] = v
      label_list[batch_idx][doc_idx] = label

  tf.compat.v1.logging.info("Number of queries: {}".format(len(qid_to_index)))
  tf.compat.v1.logging.info(
      "Number of documents in total: {}".format(total_docs))
  tf.compat.v1.logging.info(
      "Number of documents discarded: {}".format(discarded_docs))

  # Convert everything to np.array.
  for k in feature_map:
    feature_map[k] = np.array(feature_map[k])
  return feature_map, np.array(label_list)


def get_train_inputs(features, labels, batch_size):
  """Set up training input in batches."""
  iterator_initializer_hook = IteratorInitializerHook()

  def _train_input_fn():
    """Defines training input fn."""
    features_placeholder = {
        k: tf.compat.v1.placeholder(v.dtype, v.shape)
        for k, v in six.iteritems(features)
    }
    if _use_multi_head():
      placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
      labels_placeholder = {
          _PRIMARY_HEAD: placeholder,
          _SECONDARY_HEAD: placeholder,
      }
    else:
      labels_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices(
        (features_placeholder, labels_placeholder))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    if _use_multi_head():
      feed_dict = {
          labels_placeholder[head_name]: labels
          for head_name in labels_placeholder
      }
    else:
      feed_dict = {labels_placeholder: labels}
    feed_dict.update(
        {features_placeholder[k]: features[k] for k in features_placeholder})
    iterator_initializer_hook.iterator_initializer_fn = (
        lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
    return iterator.get_next()

  return _train_input_fn, iterator_initializer_hook


def get_eval_inputs(features, labels):
  """Set up eval inputs in a single batch."""
  iterator_initializer_hook = IteratorInitializerHook()

  def _eval_input_fn():
    """Defines eval input fn."""
    features_placeholder = {
        k: tf.compat.v1.placeholder(v.dtype, v.shape)
        for k, v in six.iteritems(features)
    }
    if _use_multi_head():
      placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
      labels_placeholder = {
          _PRIMARY_HEAD: placeholder,
          _SECONDARY_HEAD: placeholder,
      }
    else:
      labels_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensors(
        (features_placeholder, labels_placeholder))
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    if _use_multi_head():
      feed_dict = {
          labels_placeholder[head_name]: labels
          for head_name in labels_placeholder
      }
    else:
      feed_dict = {labels_placeholder: labels}
    feed_dict.update(
        {features_placeholder[k]: features[k] for k in features_placeholder})
    iterator_initializer_hook.iterator_initializer_fn = (
        lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
    return iterator.get_next()

  return _eval_input_fn, iterator_initializer_hook


def make_serving_input_fn():
  """Returns serving input fn to receive tf.Example."""
  feature_spec = tf.feature_column.make_parse_example_spec(
      example_feature_columns().values())
  return tf.estimator.export.build_parsing_serving_input_receiver_fn(
      feature_spec)


def make_transform_fn():
  """Returns a transform_fn that converts features to dense Tensors."""

  def _transform_fn(features, mode):
    """Defines transform_fn."""
    if mode == tf.estimator.ModeKeys.PREDICT:
      # We expect tf.Example as input during serving. In this case, group_size
      # must be set to 1.
      if FLAGS.group_size != 1:
        raise ValueError(
            "group_size should be 1 to be able to export model, but get %s" %
            FLAGS.group_size)
      context_features, example_features = (
          tfr.feature.encode_pointwise_features(
              features=features,
              context_feature_columns=None,
              example_feature_columns=example_feature_columns(),
              mode=mode,
              scope="transform_layer"))
    else:
      context_features, example_features = tfr.feature.encode_listwise_features(
          features=features,
          context_feature_columns=None,
          example_feature_columns=example_feature_columns(),
          mode=mode,
          scope="transform_layer")

    return context_features, example_features

  return _transform_fn


def make_score_fn():
  """Returns a groupwise score fn to build `EstimatorSpec`."""

  def _score_fn(unused_context_features, group_features, mode, unused_params,
                unused_config):
    """Defines the network to score a group of documents."""
    with tf.compat.v1.name_scope("input_layer"):
      group_input = [
          tf.compat.v1.layers.flatten(group_features[name])
          for name in sorted(example_feature_columns())
      ]
      input_layer = tf.concat(group_input, 1)
      tf.compat.v1.summary.scalar("input_sparsity",
                                  tf.nn.zero_fraction(input_layer))
      tf.compat.v1.summary.scalar("input_max",
                                  tf.reduce_max(input_tensor=input_layer))
      tf.compat.v1.summary.scalar("input_min",
                                  tf.reduce_min(input_tensor=input_layer))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    cur_layer = tf.compat.v1.layers.batch_normalization(
        input_layer, training=is_training)
    for i, layer_width in enumerate(int(d) for d in FLAGS.hidden_layer_dims):
      cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
      cur_layer = tf.compat.v1.layers.batch_normalization(
          cur_layer, training=is_training)
      cur_layer = tf.nn.relu(cur_layer)
      tf.compat.v1.summary.scalar("fully_connected_{}_sparsity".format(i),
                                  tf.nn.zero_fraction(cur_layer))
    cur_layer = tf.compat.v1.layers.dropout(
        cur_layer, rate=FLAGS.dropout_rate, training=is_training)
    logits = tf.compat.v1.layers.dense(cur_layer, units=FLAGS.group_size)
    if _use_multi_head():
      # Duplicate the logits for both heads.
      return {_PRIMARY_HEAD: logits, _SECONDARY_HEAD: logits}
    else:
      return logits

  return _score_fn


# Adding a new metric, Bilingual Lexical Induction

#class BLIMetric(_RankingMetric):
#  """Implements Bilingual Lexicon Induction Metric (BLI)."""
#
#  def __init__(self, name):
#    """Constructor."""
#    self._name = name
#
#  @property
#  def name(self):
#    """The metric name."""
#    return self._name
#
#  def compute(self, labels, predictions, weights):
#    """See `_RankingMetric`."""
#    list_size = tf.shape(input=predictions)[1]
#    labels, predictions, weights, topn = _prepare_and_validate_params(
#        labels, predictions, weights, list_size)
#    sorted_labels = utils.sort_by_scores(predictions, [labels])[0]
#    # Relevance = 1.0 when labels = 2.0.
#    relevance = tf.cast(tf.equal(sorted_labels, 2.0), dtype=tf.float32)
#    #We only consider the first suggestion [:,0] and BLI has a shape of [batch_size, 1].
#    bli = tf.reshape(relevance[:,0],(list_size,1))
#
#    per_list_weights = _per_example_weights_to_per_list_weights(
#        weights=weights,
#        relevance=tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32))
#    return bli, per_list_weights
#
#
#
#  def _mean_reciprocal_rank_fn(labels, predictions, features):
#    """Returns mean reciprocal rank as the metric."""
#    return mean_reciprocal_rank(
#        labels,
#        predictions,
#        weights=_get_weights(features),
#        topn=topn,
#        name=name)
#
#
#def bilingual_lexical_induction(labels,
#                                predictions,
#                                weights=None,
#                                name=None):
#  """Computes Bilingual Lexicon Induction (BLI).
#  Args:
#    labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
#      relevant example.
#    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
#      the ranking score of the corresponding example.
#    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
#      former case is per-example and the latter case is per-list.
#    name: A string used as the name for this metric.
#  Returns:
#    A metric for the weighted bilingual lexical induction of the batch.
#  """
#  metric = BLIMetric(name)
#  with tf.compat.v1.name_scope(metric.name, 'bilingual_lexicon_induction',
#                               (labels, predictions, weights)):
#    bli, per_list_weights = metric.compute(labels, predictions, weights)
#    return tf.compat.v1.metrics.mean(bli, per_list_weights)

def _get_weights(features):
    """Get weights tensor from features and reshape it to 2-D if necessary."""
    weights = None
    return weights

def _per_example_weights_to_per_list_weights(weights, relevance):
  """Computes per list weight from per example weight.
  The per-list weights are computed as:
    per_list_weights = sum(weights * relevance) / sum(relevance).
  For the list with sum(relevance) = 0, we set a default weight as the following
  average weight:
    sum(per_list_weights) / num(sum(relevance) != 0)
  Such a computation is good for the following scenarios:
    - When all the weights are 1.0, the per list weights will be 1.0 everywhere,
      even for lists without any relevant examples because
        sum(per_list_weights) ==  num(sum(relevance) != 0)
      This handles the standard ranking metrics where the weights are all 1.0.
    - When every list has a nonzero weight, the default weight is not used. This
      handles the unbiased metrics well.
    - For the mixture of the above 2 scenario, the weights for lists with
      nonzero relevance is proportional to
        per_list_weights / sum(per_list_weights) *
        num(sum(relevance) != 0) / num(lists).
      The rest have weights 1.0 / num(lists).
  Args:
    weights:  The weights `Tensor` of shape [batch_size, list_size].
    relevance:  The relevance `Tensor` of shape [batch_size, list_size].
  Returns:
    The per list `Tensor` of shape [batch_size, 1]
  """
  per_list_relevance = tf.reduce_sum(
      input_tensor=relevance, axis=1, keepdims=True)
  nonzero_relevance = tf.cast(tf.greater(per_list_relevance, 0.0), tf.float32)
  nonzero_relevance_count = tf.reduce_sum(
      input_tensor=nonzero_relevance, axis=0, keepdims=True)

  per_list_weights = tf.compat.v1.math.divide_no_nan(
      tf.reduce_sum(input_tensor=weights * relevance, axis=1, keepdims=True),
      per_list_relevance)
  sum_weights = tf.reduce_sum(
      input_tensor=per_list_weights, axis=0, keepdims=True)

  avg_weight = tf.compat.v1.math.divide_no_nan(sum_weights,
                                               nonzero_relevance_count)
  return tf.compat.v1.where(
      tf.greater(per_list_relevance, 0.0), per_list_weights,
      tf.ones_like(per_list_weights) * avg_weight)

def _prepare_and_validate_params(labels, predictions, weights=None, topn=None):
  """Prepares and validates the parameters.
  Args:
    labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
      relevant example.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    topn: A cutoff for how many examples to consider for this metric.
  Returns:
    (labels, predictions, weights, topn) ready to be used for metric
    calculation.
  """
  labels = tf.convert_to_tensor(value=labels)
  predictions = tf.convert_to_tensor(value=predictions)
  weights = 1.0 if weights is None else tf.convert_to_tensor(value=weights)
  example_weights = tf.ones_like(labels) * weights
  predictions.get_shape().assert_is_compatible_with(example_weights.get_shape())
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  predictions.get_shape().assert_has_rank(2)
  if topn is None:
    topn = tf.shape(input=predictions)[1]

  # All labels should be >= 0. Invalid entries are reset.
  is_label_valid = utils.is_label_valid(labels)
  labels = tf.compat.v1.where(is_label_valid, labels, tf.zeros_like(labels))
  predictions = tf.compat.v1.where(
      is_label_valid, predictions, -1e-6 * tf.ones_like(predictions) +
      tf.reduce_min(input_tensor=predictions, axis=1, keepdims=True))
  return labels, predictions, example_weights, topn

# We need the relevance of the correct vocabulary
if flags.query_extraction == 'binary':
    ground_truth = 1
else:
    ground_truth = flags.query_size

def bilingual_lexical_induction(labels,
                                predictions,
                                features):
  """Computes Bilingual Lexicon Induction (BLI).
  Args:
    labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
      relevant example.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    name: A string used as the name for this metric.
  Returns:
    A metric for the weighted bilingual lexical induction of the batch.
  """
  weights=_get_weights(features)
  labels, predictions, weights,_ = _prepare_and_validate_params(
        labels, predictions, weights)
  sorted_labels = utils.sort_by_scores(predictions, [labels])[0]
  # Relevance = 1.0 when labels = ground_truth
  relevance = tf.cast(tf.equal(sorted_labels, ground_truth), dtype=tf.float32)
  #We only consider the first suggestion [:,0] and BLI has a shape of [batch_size, 1].
  list_size=tf.shape(input=relevance)[0]
  bli = tf.reshape(relevance[:,0],(list_size,1))
  per_list_weights = _per_example_weights_to_per_list_weights(
        weights=weights,
        relevance=tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32))
  return tf.compat.v1.metrics.mean(bli, per_list_weights)


def get_eval_metric_fns():
  """Returns a dict from name to metric functions."""
  metric_fns = {}
  metric_fns.update({
      "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
          tfr.metrics.RankingMetricKey.ARP,
          tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY,
      ]
  })
  metric_fns.update({
      "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
          tfr.metrics.RankingMetricKey.NDCG, topn=topn)
      for topn in [1, 3, 5, 10]
  })
  #Adding the new metric
  metric_fns.update({
      "metric/bli": bilingual_lexical_induction
  })
    
  return metric_fns


def train_and_eval():
  """Train and Evaluate."""

  features, labels = load_libsvm_data(FLAGS.train_path, FLAGS.list_size)
  train_input_fn, train_hook = get_train_inputs(features, labels,
                                                FLAGS.train_batch_size)

  features_vali, labels_vali = load_libsvm_data(FLAGS.vali_path,
                                                FLAGS.list_size)
  vali_input_fn, vali_hook = get_eval_inputs(features_vali, labels_vali)

  features_test, labels_test = load_libsvm_data(FLAGS.test_path,
                                                FLAGS.list_size)
  test_input_fn, test_hook = get_eval_inputs(features_test, labels_test)

  optimizer = tf.compat.v1.train.AdagradOptimizer(
      learning_rate=FLAGS.learning_rate)

  def _train_op_fn(loss):
    """Defines train op used in ranking head."""
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    minimize_op = optimizer.minimize(
        loss=loss, global_step=tf.compat.v1.train.get_global_step())
    train_op = tf.group([minimize_op, update_ops])
    return train_op

  if _use_multi_head():
    primary_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(FLAGS.loss),
        eval_metric_fns=get_eval_metric_fns(),
        train_op_fn=_train_op_fn,
        name=_PRIMARY_HEAD)
    secondary_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(FLAGS.secondary_loss),
        eval_metric_fns=get_eval_metric_fns(),
        train_op_fn=_train_op_fn,
        name=_SECONDARY_HEAD)
    ranking_head = tfr.head.create_multi_ranking_head(
        [primary_head, secondary_head], [1.0, FLAGS.secondary_loss_weight])
  else:
    ranking_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(FLAGS.loss),
        eval_metric_fns=get_eval_metric_fns(),
        train_op_fn=_train_op_fn)

  estimator = tf.estimator.Estimator(
      model_fn=tfr.model.make_groupwise_ranking_fn(
          group_score_fn=make_score_fn(),
          group_size=FLAGS.group_size,
          transform_fn=make_transform_fn(),
          ranking_head=ranking_head),
      config=tf.estimator.RunConfig(
          FLAGS.output_dir, save_checkpoints_steps=1000))

  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      hooks=[train_hook],
      max_steps=FLAGS.num_train_steps)
  # Export model to accept tf.Example when group_size = 1.
  if FLAGS.group_size == 1:
    vali_spec = tf.estimator.EvalSpec(
        input_fn=vali_input_fn,
        hooks=[vali_hook],
        steps=1,
        exporters=tf.estimator.LatestExporter(
            "latest_exporter",
            serving_input_receiver_fn=make_serving_input_fn()),
        start_delay_secs=0,
        throttle_secs=30)
  else:
    vali_spec = tf.estimator.EvalSpec(
        input_fn=vali_input_fn,
        hooks=[vali_hook],
        steps=1,
        start_delay_secs=0,
        throttle_secs=30)

  # Train and validate
  tf.estimator.train_and_evaluate(estimator, train_spec, vali_spec)

  # Evaluate on the test data.
  estimator.evaluate(input_fn=test_input_fn, hooks=[test_hook])


def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  train_and_eval()


if __name__ == "__main__":
  flags.mark_flag_as_required("train_path")
  flags.mark_flag_as_required("vali_path")
  flags.mark_flag_as_required("test_path")
  flags.mark_flag_as_required("output_dir")

  tf.compat.v1.app.run()