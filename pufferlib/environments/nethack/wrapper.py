"""Taken & adapted from Chaos Dwarf in Nethack Challenge Starter Kit:
https://github.com/Miffyli/nle-sample-factory-baseline


MIT License

Copyright (c) 2021 Anssi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os

import cv2
import gym
import numpy as np
from numba import njit
from nle import nethack
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

#import render_utils

SMALL_FONT_PATH = os.path.join(__package__.replace(".", "/"), "Hack-Regular.ttf")

# Mapping of 0-15 colors used.
# Taken from bottom image here. It seems about right
# https://i.stack.imgur.com/UQVe5.png
COLORS = [
    "#000000",
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#808080",  # - flipped these ones around
    "#C0C0C0",  # | the gray-out dull stuff
    "#FF0000",
    "#00FF00",
    "#FFFF00",
    "#0000FF",
    "#FF00FF",
    "#00FFFF",
    "#FFFFFF",
]


@njit
def _tile_characters_to_image(
    out_image,
    chars,
    colors,
    output_height_chars,
    output_width_chars,
    char_array,
    offset_h,
    offset_w,
):
    """
    Build an image using cached images of characters in char_array to out_image
    """
    char_height = char_array.shape[3]
    char_width = char_array.shape[4]
    for h in range(output_height_chars):
        h_char = h + offset_h
        # Stuff outside boundaries is not visible, so
        # just leave it black
        if h_char < 0 or h_char >= chars.shape[0]:
            continue
        for w in range(output_width_chars):
            w_char = w + offset_w
            if w_char < 0 or w_char >= chars.shape[1]:
                continue
            char = chars[h_char, w_char]
            color = colors[h_char, w_char]
            h_pixel = h * char_height
            w_pixel = w * char_width
            out_image[
                :, h_pixel : h_pixel + char_height, w_pixel : w_pixel + char_width
            ] = char_array[char, color]


def _initialize_char_array(font_size, rescale_font_size):
    """Draw all characters in PIL and cache them in numpy arrays

    if rescale_font_size is given, assume it is (width, height)

    Returns a np array of (num_chars, num_colors, char_height, char_width, 3)
    """
    font = ImageFont.truetype(SMALL_FONT_PATH, font_size)
    dummy_text = "".join(
        [(chr(i) if chr(i).isprintable() else " ") for i in range(256)]
    )
    _, _, image_width, image_height = font.getbbox(dummy_text)
    # Above can not be trusted (or its siblings)....
    image_width = int(np.ceil(image_width / 256) * 256)

    char_width = rescale_font_size[0]
    char_height = rescale_font_size[1]

    char_array = np.zeros((256, 16, char_height, char_width, 3), dtype=np.uint8)
    image = Image.new("RGB", (image_width, image_height))
    image_draw = ImageDraw.Draw(image)
    for color_index in range(16):
        image_draw.rectangle((0, 0, image_width, image_height), fill=(0, 0, 0))
        image_draw.text((0, 0), dummy_text, fill=COLORS[color_index], spacing=0)

        arr = np.array(image).copy()
        arrs = np.array_split(arr, 256, axis=1)
        for char_index in range(256):
            char = arrs[char_index]
            if rescale_font_size:
                char = cv2.resize(char, rescale_font_size, interpolation=cv2.INTER_AREA)
            char_array[char_index, color_index] = char
    return char_array


class RenderCharImagesWithNumpyWrapper(gym.Wrapper):
    """
    Render characters as images, using PIL to render characters like we humans see on screen
    but then some caching and numpy stuff to speed up things.

    To speed things up, crop image around the player.
    """

    def __init__(
        self,
        env,
        font_size=9,
        crop_size=12,
        rescale_font_size=(6, 6),
        blstats_cursor=False,
    ):
        super().__init__(env)
        self.char_array = _initialize_char_array(font_size, rescale_font_size)
        self.char_height = self.char_array.shape[2]
        self.char_width = self.char_array.shape[3]
        # Transpose for CHW
        self.char_array = self.char_array.transpose(0, 1, 4, 2, 3)

        self.crop_size = crop_size
        self.blstats_cursor = blstats_cursor

        self.half_crop_size = crop_size // 2
        self.output_height_chars = crop_size
        self.output_width_chars = crop_size
        self.chw_image_shape = (
            3,
            self.output_height_chars * self.char_height,
            self.output_width_chars * self.char_width,
        )

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.chw_image_shape, dtype=np.uint8
        )
 
        '''
        obs_spaces = {
            "screen_image": gym.spaces.Box(
                low=0, high=255, shape=self.chw_image_shape, dtype=np.uint8
            )
        }
        obs_spaces.update(
            [
                (k, self.env.observation_space[k])
                for k in self.env.observation_space
                if k not in ["tty_chars", "tty_colors"]
            ]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)
        '''

        self.render_mode = 'rgb_array'

    def _render_text_to_image(self, obs):
        chars = obs["tty_chars"]
        colors = obs["tty_colors"]
        offset_w = 0
        offset_h = 0
        if self.crop_size:
            # Center around player
            if self.blstats_cursor:
                center_x, center_y = obs["blstats"][:2]
            else:
                center_y, center_x = obs["tty_cursor"]
            offset_h = center_y - self.half_crop_size
            offset_w = center_x - self.half_crop_size

        out_image = np.zeros(self.chw_image_shape, dtype=np.uint8)

        _tile_characters_to_image(
            out_image=out_image,
            chars=chars,
            colors=colors,
            output_height_chars=self.output_height_chars,
            output_width_chars=self.output_width_chars,
            char_array=self.char_array,
            offset_h=offset_h,
            offset_w=offset_w,
        )

        return out_image
        obs["screen_image"] = out_image
        del obs["tty_chars"]
        del obs["tty_colors"]
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs = obs
        obs = self._render_text_to_image(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.obs = obs
        obs = self._render_text_to_image(obs)
        return obs

    def render(self, mode='rgb_array'):
        return self.obs


class RenderCharImagesWithNumpyWrapperV2(gym.Wrapper):
    """
    Same as V1, but simpler and faster.
    """

    def __init__(
        self,
        env,
        font_size=9,
        crop_size=12,
        rescale_font_size=(6, 6),
    ):
        super().__init__(env)
        self.char_array = _initialize_char_array(font_size, rescale_font_size)
        self.char_height = self.char_array.shape[2]
        self.char_width = self.char_array.shape[3]
        # Transpose for CHW
        self.char_array = self.char_array.transpose(0, 1, 4, 2, 3)
        self.char_array = np.ascontiguousarray(self.char_array)
        self.crop_size = crop_size

        crop_rows = crop_size or nethack.nethack.TERMINAL_SHAPE[0]
        crop_cols = crop_size or nethack.nethack.TERMINAL_SHAPE[1]

        self.chw_image_shape = (
            3,
            crop_rows * self.char_height,
            crop_cols * self.char_width,
        )

        obs_spaces = {
            "screen_image": gym.spaces.Box(
                low=0, high=255, shape=self.chw_image_shape, dtype=np.uint8
            )
        }
        obs_spaces.update(
            [
                (k, self.env.observation_space[k])
                for k in self.env.observation_space
                if k not in ["tty_chars", "tty_colors"]
            ]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _populate_obs(self, obs):
        screen = np.zeros(self.chw_image_shape, order="C", dtype=np.uint8)
        render_utils.render_crop(
            obs["tty_chars"],
            obs["tty_colors"],
            obs["tty_cursor"],
            self.char_array,
            screen,
            crop_size=self.crop_size,
        )
        obs["screen_image"] = screen

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._populate_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._populate_obs(obs)
        return obs
