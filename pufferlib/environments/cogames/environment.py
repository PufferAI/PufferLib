"""CoGames wrapper for PufferLib."""

import functools
from cogames.cli.mission import get_mission
from mettagrid import PufferMettaGridEnv
from mettagrid.envs.stats_tracker import StatsTracker
from mettagrid.simulator import Simulator
from mettagrid.util.stats_writer import NoopStatsWriter


def env_creator(name="cogames.cogs_v_clips.machina_1.open_world"):
    return functools.partial(make, name=name)


def make(name="cogames.cogs_v_clips.machina_1.open_world", variants=None, cogs=None, render_mode="auto", seed=None, buf=None):
    mission_name = name.removeprefix("cogames.cogs_v_clips.") if name.startswith("cogames.cogs_v_clips.") else name
    variants = variants.split() if isinstance(variants, str) else variants
    _, env_cfg, _ = get_mission(mission_name, variants_arg=variants, cogs=cogs)

    render = "none" if render_mode == "auto" else "unicode" if render_mode in {"human", "ansi"} else render_mode
    simulator = Simulator()
    simulator.add_event_handler(StatsTracker(NoopStatsWriter()))
    env = PufferMettaGridEnv(simulator=simulator, cfg=env_cfg, buf=buf, seed=seed or 0)
    env.render_mode = render
    if seed:
        env.reset(seed)
    return env
