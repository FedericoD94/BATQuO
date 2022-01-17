import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import numpy as np

dtype = np.float32
true_mean = dtype([0, 0])
true_cov = dtype([[1, 0.5], [0.5, 1]])
num_results = 500
num_chains = 50

# Target distribution is defined through the Cholesky decomposition
chol = tf.linalg.cholesky(true_cov)
target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

# Initial state of the chain
init_state = np.ones([num_chains, 2], dtype=dtype)

# Run Slice Samper for `num_results` iterations for `num_chains`
# independent chains:
@tf.function
def run_mcmc():
  states = tfp.mcmc.sample_chain(
      num_results=num_results,
      current_state=init_state,
      kernel=tfp.mcmc.SliceSampler(
          target_log_prob_fn=target.log_prob,
          step_size=1.0,
          max_doublings=5),
      num_burnin_steps=200,
      num_steps_between_results=1,
      trace_fn=None,
      seed=47)
  return states

states = run_mcmc()

sample_mean = tf.reduce_mean(states, axis=[0, 1])
z = (states - sample_mean)[..., tf.newaxis]
sample_cov = tf.reduce_mean(
    tf.matmul(z, tf.transpose(z, [0, 1, 3, 2])), [0, 1])

print('sample mean', sample_mean.numpy())
print('sample covariance matrix', sample_cov.numpy())