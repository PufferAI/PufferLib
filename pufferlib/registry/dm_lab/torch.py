import pufferlib.models


class Policy(pufferlib.models.Convolutional):
    def __init__(
            self,
            envs,
            flat_size=3136,
            channels_last=True,
            downsample=1,
            input_size=512,
            hidden_size=128,
            output_size=128,
            **kwargs
        ):
        super().__init__(
            envs,
            framestack=3,
            flat_size=flat_size,
            channels_last=channels_last,
            downsample=downsample,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            **kwargs
        )
