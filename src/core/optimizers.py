from rl.util import AdditionalUpdatesOptimizer as KerasAdditionalUpdatesOptimizer

###


class AdditionalUpdatesOptimizer(KerasAdditionalUpdatesOptimizer):
    def get_config(self):
        return self.optimizer.get_config()

    def _create_slots(self, *args, **kwargs):
        return self.optimizer._create_slots(*args, **kwargs)

    def _prepare_local(self, *args, **kwargs):
        return self.optimizer._prepare_local(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        return self.optimizer.set_weights(*args, **kwargs)

    def _resource_apply_dense(self, *args, **kwargs):
        return self.optimizer._resource_apply_dense(*args, **kwargs)

    def _resource_apply_sparse(self, *args, **kwargs):
        return self.optimizer._resource_apply_sparse(*args, **kwargs)
