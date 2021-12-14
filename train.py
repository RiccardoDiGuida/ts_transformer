import tensorflow as tf
from transformer import Transformer
import time


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def train(ds):
    num_layers = 4
    d_model = ds.element_spec[1].shape[1]
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    loss_object = tf.keras.losses.MeanSquaredError(reduction='auto')

    def loss_function(real, pred):
        loss_ = loss_object(real, pred)
        return tf.reduce_sum(loss_)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    transformer = Transformer(num_layers=num_layers,
                              d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,
                              pe_input=100,
                              pe_target=100,
                              rate=dropout_rate)

    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    @tf.function()
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp],
                                         training=True)
            loss = loss_function(tar_real, predictions)
            # print(f"inp: {inp} \n tar_inp: {tar_inp} \n tar_real: {tar_real} \n loss: {loss}")

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    train_batches = ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    EPOCHS = 200
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()

        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f}')

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')