from models.base import BaseModel

###

GAMMA = 0.95

MEMORY_WINDOW_LENGTH = 1  # TODO: is this right?
BATCH_SIZE = MEMORY_WINDOW_LENGTH + 2

###


class DQN(BaseModel):
    def _train(self):
        if self.memory.nb_entries < BATCH_SIZE + 2:
            return False

        # return number of {BATCH_SIZE} samples in random order.
        samples = self.memory.sample(batch_size=BATCH_SIZE)

        for sample in samples:
            observations_dict, action, reward, finished, _ = sample
            observation = observations_dict[0]['network_input']
            observation = observation.astype('float32')

            target = self.target_network.predict(observation)

            if finished is True:
                target[0][action] = reward
            else:
                q_future_value = max(self.target_network.predict(observation)[0])
                target[0][action] = reward + q_future_value * GAMMA

            self.network.fit([observation], target, epochs=1, verbose=1)

        return True
