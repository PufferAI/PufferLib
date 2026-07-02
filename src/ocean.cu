// Custom CUDA encoder registry for ocean environments.
// Each env's encoder lives in its own translation-unit include below; this file
// only wires them into the Encoder vtable via create_custom_encoder().
// Included by pufferlib.cu — requires precision_t, PrecisionTensor, Allocator, puf_mm, etc.

#include "nmmo3.cu"
#include "craftax.cu"

// Override encoder vtable for known ocean environments. No-op for unknown envs.
static void create_custom_encoder(const std::string& env_name, Encoder* enc) {
    if (env_name == "craftax") {
        *enc = Encoder{
            .forward = craftax_encoder_forward,
            .backward = craftax_encoder_backward,
            .init_weights = craftax_encoder_init_weights,
            .reg_params = craftax_encoder_reg_params,
            .reg_train = craftax_encoder_reg_train,
            .reg_rollout = craftax_encoder_reg_rollout,
            .create_weights = craftax_encoder_create_weights,
            .free_weights = craftax_encoder_free_weights,
            .free_activations = craftax_encoder_free_activations,
            .in_dim = enc->in_dim, .out_dim = enc->out_dim,
            .activation_size = sizeof(CraftaxEncoderActivations),
        };
        return;
    }
    if (env_name == "nmmo3") {
        *enc = Encoder{
            .forward = nmmo3_encoder_forward,
            .backward = nmmo3_encoder_backward,
            .init_weights = nmmo3_encoder_init_weights,
            .reg_params = nmmo3_encoder_reg_params,
            .reg_train = nmmo3_encoder_reg_train,
            .reg_rollout = nmmo3_encoder_reg_rollout,
            .create_weights = nmmo3_encoder_create_weights,
            .free_weights = nmmo3_encoder_free_weights,
            .free_activations = nmmo3_encoder_free_activations,
            .in_dim = enc->in_dim, .out_dim = enc->out_dim,
            .activation_size = sizeof(NMMO3EncoderActivations),
        };
    }
}
