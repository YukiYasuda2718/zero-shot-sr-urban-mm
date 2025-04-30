from logging import getLogger

from src.models.neural_nets.base_config import BaseModelConfig
from src.models.neural_nets.sesrcnn import SESRCNN
from src.models.neural_nets.sesrcnn_with_pos import SESRCNNWithPos
from src.models.neural_nets.sr_neural_operator_v01 import SRNeuralOperatorV01
from src.models.neural_nets.sr_neural_operator_v02 import SRNeuralOperatorV02

logger = getLogger()


def make_model(config: BaseModelConfig):
    if config.model_name == "SESRCNN":
        logger.info("SESRCNN is being created.")
        return SESRCNN(config)
    elif config.model_name == "SESRCNNWithPos":
        logger.info("SESRCNNWithPos is being created.")
        return SESRCNNWithPos(config)
    elif config.model_name == "SRNeuralOperatorV01":
        logger.info("SRNeuralOperatorV01 is being created.")
        return SRNeuralOperatorV01(config)
    elif config.model_name == "SRNeuralOperatorV02":
        logger.info("SRNeuralOperatorV02 is being created.")
        return SRNeuralOperatorV02(config)
    else:
        raise Exception()
