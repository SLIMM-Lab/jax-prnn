import numpy as np
import jax
import jax.numpy as jnp
import optax
import copy
from flax.training import train_state


class Trainer:
    """Class for handling PRNN training tasks using JAX.

    Wraps a Flax model and performs training and evaluation tasks.
    Loss function and optimizer are configurable. Early stopping is
    implemented with adjustable patience.
    """

    def __init__(self, model, params, **kwargs):
        self._model = model
        self._epoch = 0
        self._criterion = kwargs.get('loss', self.mse_loss)

        # Get initial parameters
        self.rng = kwargs.get('random_key', jax.random.PRNGKey(0))
        self.rng, init_rng = jax.random.split(self.rng)

        # Get PRNN material properties
        self.material = kwargs.get('material', None)

        # Print parameter count
        total_params = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
        print('Total parameter count:', total_params)

        max_epochs = kwargs.get('max_epochs', 10000)
        self.train_losses = np.zeros(max_epochs)
        self.val_losses = np.zeros(max_epochs)

        lr = kwargs.get('learning_rate', 1e-3)
        self._optimizer = optax.adam(lr)

        # Initialize train state
        self._state = train_state.TrainState.create(
            apply_fn=self._model.apply,
            params={'params': params['params']},
            tx=self._optimizer)

    @staticmethod
    def mse_loss(preds, targets):
        """Mean squared error loss function"""
        # NOTE: this function is separately defined in the jit train & eval train step functions
        return jnp.mean((preds - targets) ** 2)

    def train_step(self, state, batch):  # learning_rate_fn
        """Single training step"""
        return self._train_step_jit(state, batch, self.material)

    @staticmethod
    @jax.jit
    def _train_step_jit(state, batch, material):
        """JIT-compiled implementation of train step"""
        x = batch[:, 0]
        t = batch[:, 1]

        def loss_fn(params):
            y = state.apply_fn(params, x, material)
            loss = jnp.mean((y - t) ** 2)  # Using MSE directly for JIT compatibility
            return loss, y

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, y), grads = grad_fn(state.params)

        # Update parameters
        new_state = state.apply_gradients(grads=grads)

        return new_state, loss

    @staticmethod
    @jax.jit
    def _eval_step_jit(state, batch, material):
        """JIT-compiled implementation of eval step"""
        x = batch[:, 0]
        t = batch[:, 1]

        y = state.apply_fn(state.params, x, material)
        loss = jnp.mean((y - t) ** 2)

        return loss

    def train(self, training_data, validation_data, **kwargs):
        """Train the model with early stopping"""
        epochs = kwargs.get('max_epochs', 100)
        patience = kwargs.get('patience', 100)
        interval = kwargs.get('interval', 1)
        train_batch_size = kwargs.get('train_batch_size', 4)
        # val_batch_size = kwargs.get('val_batch_size', 100)
        verbose = kwargs.get('verbose', True)
        seq_length = kwargs.get('seq_length', 60)
        feature_dim = kwargs.get('feature_dim', 3)

        stall_iters = 0
        self._best_val = float('inf')
        self._best_params = None

        for i in range(epochs):
            self._epoch = i
            # --- Training loop ---
            # Note: This process could potentially be accelerated by computing batches in parallel
            # Create training batches
            shuffled_training_data = np.random.permutation(training_data)    # by default only along first axis
            training_batches = shuffled_training_data.reshape(-1, train_batch_size, 2, seq_length, feature_dim)

            # Update
            running_loss = 0.0
            for batch in training_batches:
                self._state, loss = self.train_step(self._state, batch)
                running_loss += loss

            running_loss /= len(training_batches)
            self.train_losses[i] = running_loss

            # Validation
            if i < interval or i % interval == 0:
                # --- Validation loop ---
                # Single dataset
                val_loss = self._eval_step_jit(self._state, validation_data, self.material)
                self.val_losses[i] = val_loss

                if verbose:
                    print('Epoch', self._epoch, 'training loss', running_loss, 'validation loss', val_loss, 'stall iters:', stall_iters, '/', patience )

                if self._epoch == 1 or val_loss <= self._best_val:
                    self._best_val = val_loss
                    self._best_params = copy.deepcopy(self._state.params)
                    stall_iters = 0

                else:
                    if i <= interval:
                        stall_iters += 1
                    else:
                        stall_iters += interval

                if stall_iters >= patience:
                    if verbose:
                        print('Early stopping criterion reached.')
                    break

        if verbose:
            print('End of training.')

    def get_losses(self):
        """Return training and validation losses"""
        return self.train_losses[:self._epoch], self.val_losses[:self._epoch]

    def save(self, filename):
        """Save model state to file using NumPy instead of TensorFlow"""
        import numpy as np
        from pathlib import Path

        # Convert JAX arrays to NumPy arrays
        params = jax.tree_util.tree_map(lambda x: np.array(x), self._state.params)
        best_params = jax.tree_util.tree_map(lambda x: np.array(x),
                                             self._best_params) if self._best_params is not None else None

        # Create parent directories if they don't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Create dictionary with state information
        checkpoint = {
            'epoch': self._epoch,
            'best_val': float(self._best_val),
            'params': params,
            'best_params': best_params,
        }

        # Save using NumPy's save function
        np.save(filename, checkpoint, allow_pickle=True)
        print(f"Model saved to {filename}.npy")

    def load(self, filename, set_best_params=True):
        """Load model state from NumPy file"""
        import numpy as np

        # Add .npy extension if not present
        if not filename.endswith('.npy'):
            filename = f"{filename}.npy"

        # Load the checkpoint
        checkpoint = np.load(filename, allow_pickle=True).item()

        # Update trainer state
        self._epoch = checkpoint['epoch']
        self._best_val = checkpoint['best_val']
        print(f"best_val: {self._best_val}")
        print(f"_epoch: {self._epoch}")


        if checkpoint['best_params'] is not None:
            # Convert NumPy arrays back to JAX arrays
            self._best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), checkpoint['best_params'])
            if set_best_params:
                self._state = self._state.replace(params=self._best_params)
            return self._best_params
        else:
            if set_best_params:
                print("No best parameters found in checkpoint. Using current parameters.")
            params = jax.tree_util.tree_map(lambda x: jnp.array(x), checkpoint['params'])
            self._state = self._state.replace(params=params)
            self._best_params = None
            return params
