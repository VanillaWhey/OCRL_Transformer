def initialize_task(config, env, init_sim=True):
    # Custom Environments
    from environments.foosball.foosball_selfplay import FoosballSelfPlay
    from environments.foosball.foosball_mixed_selfplay import FoosballMixedSelfPlay

    # Mappings from strings to environments
    task_map = {
        "FoosballSelfPlay": FoosballSelfPlay,
        "FoosballMixedSelfPlay": FoosballMixedSelfPlay
    }

    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    sim_config = SimConfig(config)
    cfg = sim_config.config

    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    env.set_task(
        task=task,
        sim_params=sim_config.get_physics_params(),
        backend="torch",
        init_sim=init_sim
    )

    return task


def initialize_physical_task(config, env):
    from utilities.system_interfaces.foosball.foosball_interface import FoosballInterface

    task_map = {
        "FoosballBlocking": FoosballInterface,
        "FoosballScoringIncoming": FoosballInterface,
        "FoosballScoringResting": FoosballInterface,
        "FoosballScoringRestingObstacle": FoosballInterface,
        "FoosballSelfPlay": FoosballInterface,
        "FoosballKeeperSelfPlay": FoosballInterface,
    }

    task = task_map[config["task_name"]](cfg=config)
    env.set_task(task)

    return task