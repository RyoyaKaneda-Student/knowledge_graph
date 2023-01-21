from utils.typing import ConstMeta

class LossFnName(metaclass=ConstMeta):
    CROSS_ENTROPY_LOSS = 'cross_entropy_loss'
    FOCAL_LOSS = 'focal_loss'

    @classmethod
    def ALL_LIST(cls):
        return cls.CROSS_ENTROPY_LOSS, cls.FOCAL_LOSS


def select_loss(loss_fn_name: str):
    from torch.nn import CrossEntropyLoss
    from .focal_loss import FocalLoss
    if loss_fn_name not in LossFnName.ALL_LIST():
        raise ValueError(f"The loss name {loss_fn_name} is not defined.")
    elif loss_fn_name == LossFnName.CROSS_ENTROPY_LOSS:
        return CrossEntropyLoss
    elif loss_fn_name == LossFnName.FOCAL_LOSS:
        return FocalLoss
