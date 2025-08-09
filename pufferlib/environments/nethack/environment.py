from pdb import set_trace as T

import shimmy
import gym
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments
#from .wrapper import RenderCharImagesWithNumpyWrapper

# Copyright (c) Facebook, Inc. and its affiliates.
import enum
import logging
import os
import random
import sys
import tempfile
import time
import warnings
import weakref

import gymnasium as gym
import numpy as np

from nle import nethack

logger = logging.getLogger(__name__)

DUNGEON_SHAPE = nethack.DUNGEON_SHAPE


DEFAULT_MSG_PAD = 256
DEFAULT_INV_PAD = 55
DEFAULT_INVSTR_PAD = 80

ASCII_SPACE = ord(" ")
ASCII_y = ord("y")
ASCII_n = ord("n")
ASCII_ESC = nethack.C("[")

FULL_ACTIONS = nethack.USEFUL_ACTIONS

SKIP_EXCEPTIONS = (b"eat", b"attack", b"direction?", b"pray")

NLE_SPACE_ITEMS = (
    (
        "glyphs",
        gym.spaces.Box(
            low=0, high=nethack.MAX_GLYPH, **nethack.OBSERVATION_DESC["glyphs"]
        ),
    ),
    ("chars", gym.spaces.Box(low=0, high=255, **nethack.OBSERVATION_DESC["chars"])),
    ("colors", gym.spaces.Box(low=0, high=15, **nethack.OBSERVATION_DESC["colors"])),
    (
        "specials",
        gym.spaces.Box(low=0, high=255, **nethack.OBSERVATION_DESC["specials"]),
    ),
    (
        "blstats",
        gym.spaces.Box(
            low=np.iinfo(np.int32).min,
            high=np.iinfo(np.int32).max,
            **nethack.OBSERVATION_DESC["blstats"],
        ),
    ),
    (
        "message",
        gym.spaces.Box(
            low=np.iinfo(np.uint8).min,
            high=np.iinfo(np.uint8).max,
            **nethack.OBSERVATION_DESC["message"],
        ),
    ),
    (
        "program_state",
        gym.spaces.Box(
            low=np.iinfo(np.int32).min,
            high=np.iinfo(np.int32).max,
            **nethack.OBSERVATION_DESC["program_state"],
        ),
    ),
    (
        "internal",
        gym.spaces.Box(
            low=np.iinfo(np.int32).min,
            high=np.iinfo(np.int32).max,
            **nethack.OBSERVATION_DESC["internal"],
        ),
    ),
    (
        "inv_glyphs",
        gym.spaces.Box(
            low=0,
            high=nethack.MAX_GLYPH,
            **nethack.OBSERVATION_DESC["inv_glyphs"],
        ),
    ),
    (
        "inv_strs",
        gym.spaces.Box(low=0, high=255, **nethack.OBSERVATION_DESC["inv_strs"]),
    ),
    (
        "inv_letters",
        gym.spaces.Box(low=0, high=127, **nethack.OBSERVATION_DESC["inv_letters"]),
    ),
    (
        "inv_oclasses",
        gym.spaces.Box(
            low=0,
            high=nethack.MAXOCLASSES,
            **nethack.OBSERVATION_DESC["inv_oclasses"],
        ),
    ),
    (
        "screen_descriptions",
        gym.spaces.Box(
            low=0, high=127, **nethack.OBSERVATION_DESC["screen_descriptions"]
        ),
    ),
    (
        "tty_chars",
        gym.spaces.Box(low=0, high=255, **nethack.OBSERVATION_DESC["tty_chars"]),
    ),
    (
        "tty_colors",
        gym.spaces.Box(
            low=0,
            high=31,
            **nethack.OBSERVATION_DESC["tty_colors"],
        ),
    ),
    (
        "tty_cursor",
        gym.spaces.Box(low=0, high=255, **nethack.OBSERVATION_DESC["tty_cursor"]),
    ),
    (
        "misc",
        gym.spaces.Box(
            low=np.iinfo(np.int32).min,
            high=np.iinfo(np.int32).max,
            **nethack.OBSERVATION_DESC["misc"],
        ),
    ),
)


class NLE(gym.Env):
    """Standard NetHack Learning Environment.

    Implements a gym interface around `nethack.Nethack`.


    Examples:
        >>> env = NLE()
        >>> obs, reset_info = env.reset()
        >>> obs, reward, done, truncation, info = env.step(0)
        >>> env.render()
    """

    # Gymnasium expects an fps rate > 0 for render checks
    # but NetHack doesn't have any. Set it to 42, because
    # that is always the answer to life, the universe and
    # everything.
    metadata = {"render_modes": ["human", "ansi", "full"], "render_fps": 42}

    class StepStatus(enum.IntEnum):
        """Specifies the status of the terminal state.

        Note:
            User may redefine this class in subtasks to handle / categorize
            more terminal states.

            It is highly advised that, in such cases, the enums already defined
            in this object are replicated in some way. See `nle.env.tasks` for
            examples on how to do this right.
        """

        ABORTED = -1
        RUNNING = 0
        DEATH = 1

    def __init__(
        self,
        save_ttyrec_every=0,
        savedir=None,
        character="mon-hum-neu-mal",
        max_episode_steps=5000,
        observation_keys=(
            "glyphs",
            "chars",
            "colors",
            "specials",
            "blstats",
            "message",
            "inv_glyphs",
            "inv_strs",
            "inv_letters",
            "inv_oclasses",
            "screen_descriptions",
            "tty_chars",
            "tty_colors",
            "tty_cursor",
        ),
        actions=None,
        options=None,
        wizard=False,
        allow_all_yn_questions=False,
        allow_all_modes=False,
        spawn_monsters=True,
        render_mode="human",
    ):
        """Constructs a new NLE environment.

        Args:
            save_ttyrec_every: Integer, if 0, no ttyrecs (game recordings) will
                be saved. Otherwise, save a ttyrec every Nth episode.
            savedir (str or None): Path to save ttyrecs (game recordings) into,
                if save_ttyrec_every is nonzero. If nonempty string, interpreted
                as a path to a new or existing directory.
                If "" (empty string) or None, NLE choses a unique directory name.
            character (str): name of character. Defaults to "mon-hum-neu-mal".
            max_episode_steps (int): maximum amount of steps allowed before the
                game is forcefully quit. In such cases, ``info["end_status"]``
                will be equal to ``StepStatus.ABORTED``. Defaults to 5000.
            observation_keys (list): keys to use when creating the observation.
                Defaults to all.
            actions (list): list of actions. If None, the full action space will
                be used, i.e. ``nle.nethack.ACTIONS``. Defaults to None.
            options (list): list of game options to initialize Nethack. If None,
                Nethack will be initialized with the options found in
                ``nle.nethack.NETHACKOPTIONS`. Defaults to None.
            wizard (bool): activate wizard mode. Defaults to False.
            allow_all_yn_questions (bool):
                If set to True, no y/n questions in step() are declined.
                If set to False, only elements of SKIP_EXCEPTIONS are not declined.
                Defaults to False.
            allow_all_modes (bool):
                If set to True, do not decline menus, text input or auto 'MORE'.
                If set to False, only skip click through 'MORE' on death.
            spawn_monsters: If False, disables normal NetHack behavior to randomly
                create monsters.
            render_mode (str): mode used to render the screen. One of
                "human" | "ansi" | "full".
                Defaults to "human", i.e. what a human would see playing the game.
        """
        self.character = character
        self._max_episode_steps = max_episode_steps
        self._allow_all_yn_questions = allow_all_yn_questions
        self._allow_all_modes = allow_all_modes
        self._save_ttyrec_every = save_ttyrec_every
        self.render_mode = render_mode

        if actions is None:
            actions = FULL_ACTIONS
        self.actions = actions

        self.last_observation = ()

        try:
            if not save_ttyrec_every:
                self.savedir = None
            elif savedir:
                self.savedir = os.path.abspath(savedir)
                os.makedirs(self.savedir)
            else:  # Empty savedir: We create our unique savedir inside nle_data/.
                parent_dir = os.path.join(os.getcwd(), "nle_data")
                os.makedirs(parent_dir, exist_ok=True)
                self.savedir = tempfile.mkdtemp(
                    prefix=time.strftime("%Y%m%d-%H%M%S_"), dir=parent_dir
                )
        except FileExistsError:
            logger.info("Using existing savedir: %s", self.savedir)
        else:
            if self.savedir:
                logger.info("Created savedir: %s", self.savedir)
            else:
                logger.info("Not saving any NLE data.")

        self._observation_keys = list(observation_keys)

        if "internal" in self._observation_keys:
            logger.warn(
                "The 'internal' NLE observation was requested. "
                "This might contain data that shouldn't be available to agents."
            )

        # Observations we always need.
        for key in (
            "glyphs",
            "blstats",
            "tty_chars",
            "message",
            "program_state",
            "internal",
        ):
            if key not in self._observation_keys:
                self._observation_keys.append(key)

        self._glyph_index = self._observation_keys.index("glyphs")
        self._blstats_index = self._observation_keys.index("blstats")
        self._message_index = self._observation_keys.index("message")
        self._program_state_index = self._observation_keys.index("program_state")
        self._internal_index = self._observation_keys.index("internal")

        self._original_observation_keys = observation_keys
        self._original_indices = tuple(
            self._observation_keys.index(key) for key in observation_keys
        )
        self._info = {}

        if self.savedir:
            ttyrec_version = ".ttyrec%i.bz2" % nethack.TTYREC_VERSION
            ttyrec_prefix = "nle.%i.%%i" % os.getpid()
            self._ttyrec_pattern = os.path.join(
                self.savedir, ttyrec_prefix + ttyrec_version
            )
            ttyrec = self._ttyrec_pattern % 0
            # Create an xlogfile with the same format of name.
            scoreprefix = ttyrec.replace("0" + ttyrec_version, "")
        else:
            ttyrec = None
            scoreprefix = None

        self.nethack = nethack.Nethack(
            observation_keys=self._observation_keys,
            options=options,
            playername="Agent-" + self.character,
            ttyrec=ttyrec,
            wizard=wizard,
            spawn_monsters=spawn_monsters,
            scoreprefix=scoreprefix,
        )
        self._close_nethack = weakref.finalize(self, self.nethack.close)

        self._random = random.SystemRandom()

        # -1 so that it's 0-based on first reset
        self._episode = -1

        space_dict = dict(NLE_SPACE_ITEMS)
        self.observation_space = gym.spaces.Dict(
            {key: space_dict[key] for key in observation_keys}
        )

        self.action_space = gym.spaces.Discrete(len(self.actions))

    def _get_observation(self, observation):
        return {
            key: observation[i]
            for key, i in zip(self._original_observation_keys, self._original_indices)
        }

    def _get_end_status(self, observation, done):
        return done or self._check_abort(observation)

    def _get_information(self, end_status):
        info = {}
        info["end_status"] = end_status
        info["is_ascended"] = self.nethack.how_done() == nethack.ASCENDED
        return info

    def print_action_meanings(self):
        for a_idx, a in enumerate(self.actions):
            print(a_idx, a)

    def _check_abort(self, observation):
        return self._steps >= self._max_episode_steps

    def step(self, action: int):
        """Steps the environment.

        Args:
            action (int): action integer as defined by ``self.action_space``.

        Returns:
            (dict, float, bool, dict): a tuple containing
                - (*dict*): an observation of the state; this will contain the keys
                  specified by ``self.observation_space``.
                - (*float*): a reward; see ``self._reward_fn`` to see how it is
                  specified.
                - (*bool*): True if the state is terminal, False otherwise.
                - (*bool*): True if the episode is truncated, False otherwise.
                - (*dict*): a dictionary of extra information (such as
                  `end_status`, i.e. a status info -- death, task win, etc. --
                  for the terminal state).
        """
        # Careful: By default we re-use Numpy arrays, so copy before!
        old_score = self.last_observation[self._blstats_index][nethack.NLE_BL_SCORE]

        observation, done = self.nethack.step(self.actions[action])
        truncated = self._check_abort(observation)

        # is_game_over = observation[self._program_state_index][0] == 1
        # if is_game_over or not self._allow_all_modes:
        #     observation, done = self._perform_known_steps(
        #         observation, done, exceptions=True
        #     )

        self._steps += 1

        self.last_observation = observation

        end_status = self._get_end_status(observation, done)

        reward = float(
            self._reward_fn(old_score, action, observation, end_status)
        )

        if end_status and not done:
            # Try to end the game nicely.
            self._quit_game(observation, done)
            done = True

        return (
            self._get_observation(observation),
            reward,
            done,
            truncated,
            self._info,
        )

    def _in_moveloop(self, observation):
        program_state = observation[self._program_state_index]
        return program_state[3]  # in_moveloop

    def reset(self, seed=None, options=None):
        """Resets the environment.

        Note:
            We attempt to manually navigate the first few menus so that the
            first seen state is ready to be acted upon by the user. This might
            fail in case Nethack is initialized with some uncommon options.

        Returns:
            (tuple) (Observation of the state as
                defined by `self.observation_space`,
                Extra game state information)
        """
        super().reset(seed=seed, options=options)
        self._episode += 1
        if self.savedir and self._episode % self._save_ttyrec_every == 0:
            new_ttyrec = self._ttyrec_pattern % self._episode
        else:
            new_ttyrec = None
        self.last_observation = self.nethack.reset(new_ttyrec, options=options)

        self._steps = 0
        done = False

        for _ in range(1000):
            # Get past initial phase of game. This should make sure
            # all the observations are present.
            if self._in_moveloop(self.last_observation):
                break
            # This fails if the agent picks up a scroll of scare
            # monster at the 0th turn and gets asked to name it.
            # Hence the defensive iteration above.
            # TODO: Detect this 'in_getlin' situation and handle it.
            self.last_observation, done = self.nethack.step(ASCII_SPACE)
            assert not done, "Game ended unexpectedly"
        else:
            warnings.warn(
                "Not in moveloop after 1000 tries, aborting (ttyrec: %s)." % new_ttyrec,
                stacklevel=2,
            )
            return self.reset(seed=seed, options=options)

        return self._get_observation(self.last_observation), {}

    def close(self):
        self._close_nethack()
        super().close()

    def seed(self, core=None, disp=None, reseed=False, lgen=None):
        """Sets the state of the NetHack RNGs after the next reset.

        NetHack 3.6 uses two RNGs, core and disp. This is to prevent
        RNG-manipulation by e.g. running into walls or other no-ops on the
        actual game state. This is a measure against "tool-assisted
        speedruns" (TAS). NLE can run in both NetHack's default mode and in
        TAS-friendly "no reseeding" if `reseed` is set to False, see below.

        lgen allows the user to use a different RNG to generate the levels in
        NetHack so that the levels become fixed and independent from the in-game
        choices.

        Arguments:
            core [int or None]: Seed for the core RNG. If None, chose a random
                value.
            disp [int or None]: Seed for the disp (anti-TAS) RNG. If None, chose
                a random value.
            reseed [boolean]: As an Anti-TAS (automation) measure,
                NetHack 3.6 reseeds with true randomness every now and then. This
                flag enables or disables this behavior. If set to True, trajectories
                won't be reproducible.
            lgen [int or None]: Seed for the level generator, used for RNG when
                NetHack generates new levels.

        Returns:
            [tuple] The seeds supplied, in the form (core, disp, reseed, lgen).
        """
        if core is None:
            core = self._random.randrange(sys.maxsize)
        if disp is None:
            disp = self._random.randrange(sys.maxsize)
        self.nethack.set_initial_seeds(core, disp, reseed, lgen)
        return (core, disp, reseed, lgen)

    def get_seeds(self):
        """Returns current seeds.

        Returns:
            (tuple): Current NetHack (core, disp, reseed, lgen) state.
        """
        return self.nethack.get_current_seeds()

    def render(self):
        """Renders the state of the environment."""
        mode = self.render_mode

        if mode == "human":
            obs = self.last_observation
            tty_chars = obs[self._observation_keys.index("tty_chars")]
            tty_colors = obs[self._observation_keys.index("tty_colors")]
            tty_cursor = obs[self._observation_keys.index("tty_cursor")]
            print(nethack.tty_render(tty_chars, tty_colors, tty_cursor))
            return None

        if mode == "full":
            message_index = self._observation_keys.index("message")
            message = bytes(self.last_observation[message_index])
            print(message[: message.index(b"\0")])
            try:
                inv_strs_index = self._observation_keys.index("inv_strs")
                inv_letters_index = self._observation_keys.index("inv_letters")

                inv_strs = self.last_observation[inv_strs_index]
                inv_letters = self.last_observation[inv_letters_index]
                for letter, line in zip(inv_letters, inv_strs):
                    if np.all(line == 0):
                        break
                    print(
                        letter.tobytes().decode("utf-8"), line.tobytes().decode("utf-8")
                    )
            except ValueError:  # inv_strs/letters not used.
                pass

            chars = self.last_observation[self._observation_keys.index("chars")]
            colors = self.last_observation[self._observation_keys.index("colors")]
            print(nethack.tty_render(chars, colors))
            return None

        if mode in ("ansi", "string"):  # Misnomer: This is the least ANSI of them all.
            chars = self.last_observation[self._observation_keys.index("chars")]
            # TODO: Why return a string here but print in the other branches?
            return "\n".join([line.tobytes().decode("utf-8") for line in chars])

        return "\nInvalid render mode: " + mode

    def __repr__(self):
        return "<%s>" % self.__class__.__name__

    def _is_episode_end(self, observation):
        """Returns whether the episode has ended.

        Tasks may override this method to specify different conditions, so long
        as the return value has a well defined __int__ method (e.g. booleans,
        numerical types, enum.IntEnum) and that value is part of StepStatus.

        The return value will be stored into info["end_status"].
        """
        return self.StepStatus.RUNNING

    def _reward_fn(self, old_score, action, observation, end_status):
        """Reward function. Difference between previous score and new score."""

        if observation[self._blstats_index][0] == 0:
            # Before game started and after it ended blstats are zero.
            return 0.0

        score = observation[self._blstats_index][nethack.NLE_BL_SCORE]
        return score - old_score

    def _perform_known_steps(self, observation, done, exceptions=True):
        steps = 0
        while not done:
            steps += 1
            if observation[self._internal_index][3]:  # xwaitforspace
                observation, done = self.nethack.step(ASCII_SPACE)
                continue

            internal = observation[self._internal_index]
            in_yn_function = internal[1]
            in_getlin = internal[2]

            if in_getlin:  # Game asking for a line of text. We don't do that.
                observation, done = self.nethack.step(ASCII_ESC)
                continue

            if in_yn_function:  # Game asking for a single character.
                # Note: No auto-yes to final questions thanks to the disclose option.
                if exceptions:
                    # This causes an annoying unnecessary copy...
                    msg = bytes(observation[self._message_index])
                    # Do not skip some questions to allow agent to select
                    # stuff to eat, attack, and to select directions.

                    # do not skip if all allowed or the allowed message appears
                    if self._allow_all_yn_questions or any(
                        el in msg for el in SKIP_EXCEPTIONS
                    ):
                        break

                # Otherwise, auto-decline.
                observation, done = self.nethack.step(ASCII_ESC)

            break
        if steps > 10:
            print(steps, "steps to get out of menus and windows.")
        return observation, done

    def _quit_game(self, observation, done):
        """Smoothly quit a game."""
        # Get out of menus and windows.
        # observation, done = self._perform_known_steps(
        #     observation, done, exceptions=False
        # )

        if done:
            return

        # Quit the game.
        actions = [0x80 | ord("q"), ord("y")]  # M-q y
        for a in actions:
            observation, done = self.nethack.step(a)

        # Answer final questions.
        # observation, done = self._perform_known_steps(
        #     observation, done, exceptions=False
        # )

        if not done:
            # Somehow, the above logic failed us.
            warnings.warn(
                "Warning: smooth quitting of game failed, aborting.", stacklevel=2
            )

def env_creator(name='nethack'):
    return functools.partial(make, name)

def make(name, buf=None, seed=0):
    '''NetHack binding creation function'''
    if name == 'nethack':
        name = 'NetHackScore-v0'

    nle = pufferlib.environments.try_import('nle')
    from nle.env.tasks import NetHackScore
    #env = NetHackScore(observation_keys=['blstats', 'chars'])
    env = NLE(observation_keys=['blstats', 'chars'])
    #env = RenderCharImagesWithNumpyWrapper(env)
    #env = shimmy.GymV21CompatibilityV0(env=env)
    env = NethackWrapper(env)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

class NethackWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.close = self.env.close
        self.close = self.env.close
        self.render_mode = 'ansi'

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.obs = obs
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.obs = obs
        return obs, reward, done, truncated, info

    def render(self):
        import nle
        chars = nle.nethack.tty_render(
            self.obs['tty_chars'], self.obs['tty_colors'], self.obs['tty_cursor'])
        return chars
