import jax
import jax.numpy as jnp
import flax.linen as nn
import jax_j2


class SoftLayer(nn.Module):
    """Custom decoder layer that applies softplus to weights before the linear transform."""
    n_matpts: int   # m: Number of material points
    n_outputs: int  # o: Number of output components
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        # x shape: [s, b, m * o] = [s, b, p]
        # s = sequence length
        # b = batch size
        # p = layer input: m * o

        # Create unconstrained weights parameter
        raw_weights = self.param('raw_weights',
                                 nn.initializers.he_uniform(dtype=jnp.float32),
                                 (self.n_matpts * self.n_outputs,self.n_outputs))   # [p, o]

        # Apply softplus to ensure weights are positive
        weights = nn.softplus(raw_weights)  # [p, o]

        # linear transform, essentially dot product between weights and input
        # Compute using Einstein summation convention
        output = jnp.einsum('sbp,po->sbo', x, weights)  # [s, b, o]

        if self.use_bias:
            bias = self.param('bias',
                              nn.initializers.zeros,
                              (self.n_outputs,))
            output = output + bias

        return output


class SparseNormLayer(nn.Module):
    """Custom decoder sparse layer with constrained connectivity pattern and weights.

    Each material point's output is connected with a weight to the corresponding PRNN output comonent.
    All weights are positive, and per component sum to one.
    """
    n_matpts: int   # m: Number of material points
    n_outputs: int  # o: Number of output components

    @nn.compact
    def __call__(self, x):
        # x shape: [s, b, m * o]
        s = x.shape[0]  # sequence length
        b = x.shape[1]  # batch size

        # Create weights parameter
        raw_weights = self.param('raw_weights',
                                 nn.initializers.he_uniform(dtype=jnp.float32),
                                 (self.n_matpts,self.n_outputs))

        # Apply softplus to ensure weights are positive
        weights = nn.softplus(raw_weights)  # [m, o]
        # Scale weights to sum to one component wise
        weights = weights / jnp.sum(weights, axis=0)  # [m, o]

        # # Alternative using softmax
        # weights = jax.nn.softmax(raw_weights, axis=0)   # [m, o]

        # Reshape input for sparse multiplication
        x_reshaped = x.reshape(s, b, self.n_matpts, self.n_outputs)  # [s, b, m, o]

        # Compute layer using Einstein summation convention
        # sbmo: input dimensions
        # mo: weight dimensions (broadcasts to match input)
        # -> sbo: output dimensions (sum over 'm')
        output = jnp.einsum('sbmo,mo->sbo', x_reshaped, weights)

        return output


class SparseSharedNormLayer(nn.Module):
    """Custom decoder sparse layer with constrained connectivity pattern and weights.

    Each material point's 3 outputs are connected to the 3 PRNN outputs with a single weight.
    All weights are positive and sum to one.
    """
    n_matpts: int   # m: Number of material points
    n_outputs: int  # o: Number of output components

    @nn.compact
    def __call__(self, x):
        # x shape: [s, b, m * o]
        s = x.shape[0]  # sequence length
        b = x.shape[1]  # batch size

        # Create unconstrained weights parameter
        raw_weights = self.param('raw_weights',
                                 nn.initializers.uniform(dtype=jnp.float32),    # NOTE: uniform distribution
                                 (self.n_matpts,))  # [m]

        # Apply softplus to ensure weights are positive
        weights = nn.softplus(raw_weights)  # [m]
        # Scale weights to sum to one
        weights = weights / jnp.sum(weights)  # [m]

        # Alternative using softmax
        # weights = nn.softmax(raw_weights)  # [m]

        # Reshape input for sparse multiplication
        x_reshaped = x.reshape(s, b, self.n_matpts, self.n_outputs)  # [s, b, m, o]

        # Compute layer using Einstein summation convention
        # sbmo: input dimensions
        # m: weight dimensions (broadcasts to match input)
        # -> sbo: output dimensions (sum over 'p')
        output = jnp.einsum('sbmo,m->sbo', x_reshaped, weights)  # [s, b, o]

        return output


class PRNN(nn.Module):
    """Physics-regularized neural network using JAX."""
    n_features: int     # f: number of input features (strain tensor components)
    n_outputs: int      # o: Number of output components
    n_matpts: int       # m: Number of material points
    decoder_type: str   # Type of decoder layer to use

    def setup(self):
        # Calculate total size of material layer
        self.n_latents = self.n_matpts * self.n_features

        # First linear layer (without bias) - encoder/localization
        self.encoder = nn.Dense(features=self.n_latents, use_bias=False, name="Encoder")

        # Decoder / homogenization layer
        if self.decoder_type == 'SoftLayer':
            # SoftLayer (Standard layer with softplus on weights)
            self.decoder = SoftLayer(n_matpts=self.n_matpts, n_outputs=self.n_outputs, name="Decoder", use_bias=False)
        elif self.decoder_type == 'SparseNormLayer':
            # Sparse, non-shared weights. Component-wise weights sum to one
            self.decoder = SparseNormLayer(n_matpts=self.n_matpts, n_outputs=self.n_outputs, name="Decoder")
        elif self.decoder_type == 'SparseSharedNormLayer':
            # Sparse, shared weights per material point, positive, and all weights sum to one
            self.decoder = SparseSharedNormLayer(n_matpts=self.n_matpts, n_outputs=self.n_outputs, name="Decoder")
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")


    def __call__(self, x, material):
        # x: [b, s, f]
        b = x.shape[0]  # batch size
        # s = sequence length (time steps)
        init_hist_state = jax_j2.init_history(b * self.n_matpts)

        # --- Apply Encoder BEFORE scan on all time steps ---
        # Input: x [b, s, f]
        # Output: strains [b, s, n_latents]
        strains = self.encoder(x)

        # Define a scan function for sequential processing
        def scan_fn(hist_state, strain_t):
            """
            Update the material point in batches
            Input: current state & new strain
            Output: new state & new stress
            """
            strain_batch = jnp.reshape(strain_t, (b * self.n_matpts, self.n_features))

            # Material model update
            stress_batch, new_hist_state = jax_j2.constitutive_update_batch(
                strain_batch, hist_state, material)

            stress_reshape = jnp.reshape(stress_batch, (b, self.n_latents))
            return new_hist_state, stress_reshape

        # Swap axes for scan, since we loop over first axis: [b, s, n_latents] -> [s, b, n_latents]
        strains_scannable = jnp.swapaxes(strains, 0, 1)

        # Apply scan: this loops over the time dimension of the strain tensor in a JAX-optimized way
        # Output collected_stresses: [s, b, n_latents]
        final_state, collected_stresses = jax.lax.scan(scan_fn, init_hist_state, strains_scannable)

        # Swap axes back: [s, b, n_latents] -> [b, s, n_latents]
        stresses_seq = jnp.swapaxes(collected_stresses, 0, 1)

        # --- Apply Decoder AFTER the scan on all time steps ---
        # Input: stresses_seq [b, s, expected_latents_from_stress]
        # Output: outputs [b, s, n_outputs]
        outputs = self.decoder(stresses_seq)

        return outputs


def create_prnn_model(n_features=3, n_outputs=3, n_matpts=8, random_key=jax.random.PRNGKey(0), decoder_type='SoftLayer'):
    """Create and initialize a PRNN model with material parameters."""
    # Create model
    model = PRNN(n_features=n_features, n_outputs=n_outputs, n_matpts=n_matpts, decoder_type=decoder_type)

    print(f'New PRNN: Input (strain) size {model.n_features} - Material layer size (points) {model.n_matpts} - Output (stress) size {model.n_outputs}')

    # Create material and yield config for actual use
    material = jax_j2.create_material()

    # Initialize model parameters
    params = model.init(random_key,
                       jnp.zeros((1, 1, n_features)),  # Dummy input
                       material,   # Material params
                       )

    return model, params, material