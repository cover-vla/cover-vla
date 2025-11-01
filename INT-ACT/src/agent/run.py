"""
The script to step into the training or evaluation.
Has model factory featue to select the model to train or evaluate.

"""
import os
import sys

import draccus
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from policy_server_client.websocket_policy_server import WebsocketPolicyServer

from src.agent.configuration_pipeline import TrainPipelineConfig
from src.utils.pipeline import get_class_from_path


@draccus.wrap()
def main(pipeline_cfg: TrainPipelineConfig):
    model_type = pipeline_cfg.model_cfg.type

    model_map = {
        "pi0": PI0Policy,
        }

    if pipeline_cfg.eval_cfg is None:
        # only training
        from src.agent.trainer import PI0Trainer
        trainer_map = {
            "pi0": PI0Trainer,
        }

        model_class = model_map.get(model_type, None)

        trainer_class = trainer_map.get(model_type, None)
        if trainer_class is None:
            raise ValueError(f"Model type {model_type} not supported for training.")

        trainer = trainer_class(train_cfg=pipeline_cfg, model_class=model_class)
        trainer.train()
    else:
        # evaluation
        if pipeline_cfg.eval_cfg.role == "server":

            model_class = model_map.get(model_type, None)

            from src.experiments.policies.policy_wrapper import LeRobotPolicyWrapper, MagmaPolicyWrapper, OctoPolicyWrapper, SpatialVLAPolicyWrapper
            policy_wrapper_map = {
                "pi0": LeRobotPolicyWrapper,
                "spatial-vla": SpatialVLAPolicyWrapper,
                "magma": MagmaPolicyWrapper,
                "octo": OctoPolicyWrapper,
            }
            policy_wrapper_class = policy_wrapper_map.get(model_type, None)
            policy = policy_wrapper_class(pipeline_cfg=pipeline_cfg, model_class=model_class)

            websocket_server = WebsocketPolicyServer(
                policy=policy,
                host=policy.host,
                port=policy.port,
            )
            websocket_server.serve_forever()

        elif pipeline_cfg.eval_cfg.role == "client":
            evaluator_class = get_class_from_path(pipeline_cfg.eval_cfg.simulator_path)
            evaluator = evaluator_class(pipeline_cfg=pipeline_cfg)
            evaluator.evaluate()
if __name__ == "__main__":
    main()
