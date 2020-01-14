import tensorflow as tf
from progress_bar import ProgressBar

class DeepDream(tf.Module):

    def __init__(self, model, artist_vector):
        self.model         = model
        self.artist_vector = tf.convert_to_tensor(artist_vector)
        self.artist_vector = tf.cast(self.artist_vector, tf.float32)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,None,None,3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
    ))
    def __call__(self, img, steps, learning_rate):
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                activations = self.model(img)
                loss        = tf.reduce_sum(tf.math.square(activations))
                # loss        = tf.tensordot(activations, self.artist_vector, axes = [1, 0])

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            #  Gradient ascent
            img = img + gradients * learning_rate

        return img

    def run_deep_dream_simple(self,
                              img,
                              steps = 100,
                              learning_rate = 1.0,
                              update_frequency = 5):
        updates = []
        progress_bar = ProgressBar(steps)
        progress_bar.start()

        steps_remaining = steps
        steps_done      = 0
        while steps_remaining:
            if steps_remaining>update_frequency:
                run_steps = tf.constant(update_frequency)
            else:
                run_steps = tf.constant(steps_remaining)
            steps_remaining -= run_steps
            steps_done      += run_steps

            img = self(img, run_steps, tf.constant(learning_rate))
            updates.append(img.numpy())
            progress_bar.update(steps_done)

        return img, updates
