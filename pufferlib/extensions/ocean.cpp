// ocean.cpp - Environment-specific encoder/decoder models for Ocean environments
// Separated from models.cpp for cleaner organization
// NOTE: This file is included directly into pufferlib.cpp inside namespace pufferlib

// Snake encoder: one-hot encode observations then linear
class SnakeEncoder : public Encoder {
    public:
        torch::nn::Linear linear{nullptr};
        int input;
        int hidden;
        int num_classes;

    SnakeEncoder(int input, int hidden, int num_classes = 8)
        : input(input), hidden(hidden), num_classes(num_classes) {
        linear = register_module("linear", torch::nn::Linear(
            torch::nn::LinearOptions(input * num_classes, hidden).bias(false)));
        torch::nn::init::orthogonal_(linear->weight, std::sqrt(2.0));
    }

    Tensor forward(Tensor x) override {
        // x is [B, input] with values 0-7
        int64_t B = x.size(0);
        auto target_dtype = linear->weight.dtype();
        // One-hot encode: [B, input] -> [B, input, num_classes]
        Tensor onehot = torch::one_hot(x.to(torch::kLong), num_classes).to(target_dtype);
        // Flatten: [B, input * num_classes]
        onehot = onehot.view({B, -1});
        return linear->forward(onehot);
    }
};

// G2048 encoder: embeddings + 3 linear layers with GELU
// Matches Python: value_embed(obs) + pos_embed -> flatten -> encoder MLP
class G2048Encoder : public Encoder {
    public:
        torch::nn::Embedding value_embed{nullptr};
        torch::nn::Embedding pos_embed{nullptr};
        torch::nn::Linear linear1{nullptr};
        torch::nn::Linear linear2{nullptr};
        torch::nn::Linear linear3{nullptr};
        int input;
        int hidden;
        static constexpr int embed_dim = 3;  // ceil(33^0.25) = 3
        static constexpr int num_grid_cells = 16;
        static constexpr int num_obs = num_grid_cells * embed_dim;  // 48

    G2048Encoder(int input, int hidden)
        : input(input), hidden(hidden) {
        // Embeddings for tile values and positions
        value_embed = register_module("value_embed", torch::nn::Embedding(18, embed_dim));
        pos_embed = register_module("pos_embed", torch::nn::Embedding(num_grid_cells, embed_dim));

        // Encoder MLP: num_obs -> 2*hidden -> hidden -> hidden
        linear1 = register_module("linear1", torch::nn::Linear(
            torch::nn::LinearOptions(num_obs, 2*hidden).bias(false)));
        torch::nn::init::orthogonal_(linear1->weight, std::sqrt(2.0));

        linear2 = register_module("linear2", torch::nn::Linear(
            torch::nn::LinearOptions(2*hidden, hidden).bias(false)));
        torch::nn::init::orthogonal_(linear2->weight, std::sqrt(2.0));

        linear3 = register_module("linear3", torch::nn::Linear(
            torch::nn::LinearOptions(hidden, hidden).bias(false)));
        torch::nn::init::orthogonal_(linear3->weight, std::sqrt(2.0));
    }

    Tensor forward(Tensor x) override {
        // x is (B, 16) uint8 tile values
        auto B = x.size(0);
        auto target_dtype = linear1->weight.dtype();

        // value_embed(obs) -> (B, 16, embed_dim)
        auto value_obs = value_embed->forward(x.to(torch::kLong)).to(target_dtype);

        // pos_embed.weight expanded to (B, 16, embed_dim)
        auto pos_obs = pos_embed->weight.unsqueeze(0).expand({B, num_grid_cells, embed_dim}).to(target_dtype);

        // grid_obs = (value_obs + pos_obs).flatten(1) -> (B, 48)
        auto grid_obs = (value_obs + pos_obs).flatten(1);

        // Encoder MLP
        auto h = torch::gelu(linear1->forward(grid_obs));
        h = torch::gelu(linear2->forward(h));
        h = torch::gelu(linear3->forward(h));
        return h;
    }
};

class SimpleG2048Encoder : public Encoder {
    public:
        torch::nn::Embedding value_embed{nullptr};
        torch::nn::Embedding pos_embed{nullptr};
        torch::nn::Linear linear1{nullptr};
        int input;
        int hidden;
        static constexpr int embed_dim = 3;  // ceil(33^0.25) = 3
        static constexpr int num_grid_cells = 16;
        static constexpr int num_obs = num_grid_cells * embed_dim;  // 48

    SimpleG2048Encoder(int input, int hidden)
        : input(input), hidden(hidden) {
        // Embeddings for tile values and positions
        value_embed = register_module("value_embed", torch::nn::Embedding(18, embed_dim));
        pos_embed = register_module("pos_embed", torch::nn::Embedding(num_grid_cells, embed_dim));

        linear1 = register_module("linear1", torch::nn::Linear(
            torch::nn::LinearOptions(num_obs, hidden).bias(false)));
        torch::nn::init::orthogonal_(linear1->weight, std::sqrt(2.0));
    }

    Tensor forward(Tensor x) override {
        // x is (B, 16) uint8 tile values
        auto B = x.size(0);
        auto target_dtype = linear1->weight.dtype();

        // value_embed(obs) -> (B, 16, embed_dim)
        auto value_obs = value_embed->forward(x.to(torch::kLong)).to(target_dtype);

        // pos_embed.weight expanded to (B, 16, embed_dim)
        auto pos_obs = pos_embed->weight.unsqueeze(0).expand({B, num_grid_cells, embed_dim}).to(target_dtype);

        // grid_obs = (value_obs + pos_obs).flatten(1) -> (B, 48)
        auto grid_obs = (value_obs + pos_obs).flatten(1);

        return linear1->forward(grid_obs);
    }
};

// NMMO3 encoder: Conv2d map processing + embedding for player discrete + projection
class NMMO3Encoder : public Encoder {
    public:
        // Multi-hot encoding factors and offsets
        torch::nn::Conv2d conv1{nullptr};
        torch::nn::Conv2d conv2{nullptr};
        torch::nn::Embedding player_embed{nullptr};
        torch::nn::Linear proj{nullptr};
        Tensor offsets{nullptr};
        int input;
        int hidden;

    NMMO3Encoder(int input, int hidden)
        : input(input), hidden(hidden) {
        // factors = [4, 4, 17, 5, 3, 5, 5, 5, 7, 4], sum = 59
        // Map processing: Conv2d(59, 128, 5, stride=3) -> ReLU -> Conv2d(128, 128, 3, stride=1) -> Flatten
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(59, 128, 5).stride(3).bias(true)));
        torch::nn::init::orthogonal_(conv1->weight, std::sqrt(2.0));
        torch::nn::init::constant_(conv1->bias, 0.0);

        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(128, 128, 3).stride(1).bias(true)));
        torch::nn::init::orthogonal_(conv2->weight, std::sqrt(2.0));
        torch::nn::init::constant_(conv2->bias, 0.0);

        // Player discrete encoder: Embedding(128, 32) -> Flatten
        // Input is 47 discrete values, output is 47*32 = 1504
        player_embed = register_module("player_embed", torch::nn::Embedding(128, 32));

        // Projection: Linear(1817, hidden) -> ReLU
        // Input: (B, 59, 11, 15)
        // After conv1(5, stride=3): (11-5)/3+1=3, (15-5)/3+1=4 -> (B, 128, 3, 4)
        // After conv2(3, stride=1): (3-3)/1+1=1, (4-3)/1+1=2 -> (B, 128, 1, 2)
        // Flatten: 128*1*2 = 256
        // player_discrete: 47*32 = 1504
        // player continuous (same 47 values): 47
        // reward: 10
        // Total: 256 + 1504 + 47 + 10 = 1817
        proj = register_module("proj", torch::nn::Linear(
            torch::nn::LinearOptions(1817, hidden).bias(true)));
        torch::nn::init::orthogonal_(proj->weight, std::sqrt(2.0));
        torch::nn::init::constant_(proj->bias, 0.0);

        // Register offsets buffer for multi-hot encoding
        // factors = [4, 4, 17, 5, 3, 5, 5, 5, 7, 4]
        // offsets = [0, 4, 8, 25, 30, 33, 38, 43, 48, 55]
        std::vector<int64_t> offset_vals = {0, 4, 8, 25, 30, 33, 38, 43, 48, 55};
        offsets = register_buffer("offsets",
            torch::tensor(offset_vals, torch::kInt64).view({1, 10, 1, 1}));
    }

    Tensor forward(Tensor x) override {
        int64_t B = x.size(0);
        auto device = x.device();
        auto target_dtype = conv1->weight.dtype();

        // Split observations: map (1650), player (47), reward (10)
        Tensor ob_map = x.narrow(1, 0, 11*15*10).view({B, 11, 15, 10});
        Tensor ob_player = x.narrow(1, 11*15*10, 47);
        Tensor ob_reward = x.narrow(1, 11*15*10 + 47, 10);

        // Multi-hot encoding for map
        // ob_map: (B, 11, 15, 10) -> permute to (B, 10, 11, 15)
        Tensor map_perm = ob_map.permute({0, 3, 1, 2}).to(torch::kInt64);
        // Add offsets: codes = map_perm + offsets
        Tensor codes = map_perm + offsets.to(device);

        // Create multi-hot buffer and scatter
        Tensor map_buf = torch::zeros({B, 59, 11, 15}, torch::TensorOptions().dtype(target_dtype).device(device));
        map_buf.scatter_(1, codes.to(torch::kInt32), 1.0f);

        // Conv layers
        Tensor map_out = torch::relu(conv1->forward(map_buf));
        map_out = conv2->forward(map_out);
        map_out = map_out.flatten(1);  // (B, 256)

        // Player discrete embedding
        Tensor player_discrete = player_embed->forward(ob_player.to(torch::kInt64)).to(target_dtype);
        player_discrete = player_discrete.flatten(1);  // (B, 1504)

        // Concatenate: map_out + player_discrete + player_continuous + reward
        Tensor obs = torch::cat({map_out, player_discrete, ob_player.to(target_dtype), ob_reward.to(target_dtype)}, 1);

        // Projection with ReLU
        obs = torch::relu(proj->forward(obs));
        return obs;
    }
};

// NMMO3 decoder: LayerNorm -> fused logits+value
class NMMO3Decoder : public Decoder {
    public:
        torch::nn::LayerNorm layer_norm{nullptr};
        torch::nn::Linear linear{nullptr};
        int hidden;
        int output;

    NMMO3Decoder(int hidden, int output)
        : hidden(hidden), output(output) {
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden})));

        linear = register_module("linear", torch::nn::Linear(
            torch::nn::LinearOptions(hidden, output + 1).bias(true)));
        torch::nn::init::orthogonal_(linear->weight, 0.01);
        torch::nn::init::constant_(linear->bias, 0.0);
    }

    std::tuple<Logits, Tensor> forward(Tensor h) override {
        Tensor x = layer_norm->forward(h);
        Tensor out = linear->forward(x);
        Tensor logits = out.narrow(-1, 0, output);
        Tensor value = out.narrow(-1, output, 1);
        return {Logits{logits, Tensor()}, value.squeeze(-1)};
    }
};

// Drive encoder: ego/partner/road encoders with max pooling
// Two modes:
//   use_fused_kernel=true:  FC -> Max (fused kernel, no intermediate layer)
//   use_fused_kernel=false: Linear -> LayerNorm -> Linear -> Max (original torch)
class DriveEncoder : public Encoder {
    public:
        // Ego encoder: Linear -> ReLU -> Linear (no max pooling, single point)
        torch::nn::Linear ego_linear1{nullptr};
        torch::nn::Linear ego_linear2{nullptr};

        // Road encoder weights - fused mode: single FC layer
        Tensor road_W{nullptr};
        Tensor road_b{nullptr};
        // Road encoder modules - torch mode: Linear -> LayerNorm -> Linear
        torch::nn::Linear road_linear1{nullptr};
        torch::nn::LayerNorm road_ln{nullptr};
        torch::nn::Linear road_linear2{nullptr};

        // Partner encoder weights - fused mode: single FC layer
        Tensor partner_W{nullptr};
        Tensor partner_b{nullptr};
        // Partner encoder modules - torch mode: Linear -> LayerNorm -> Linear
        torch::nn::Linear partner_linear1{nullptr};
        torch::nn::LayerNorm partner_ln{nullptr};
        torch::nn::Linear partner_linear2{nullptr};

        // Shared embedding
        torch::nn::Linear shared_linear{nullptr};
        int input;
        int hidden;
        bool use_fused_kernel;

    DriveEncoder(int input, int hidden, bool use_fused_kernel = true)
        : input(128), hidden(hidden), use_fused_kernel(use_fused_kernel) {

        // Ego encoder: 7 -> 128 -> 128 (Linear -> ReLU -> Linear)
        ego_linear1 = register_module("ego_linear1", torch::nn::Linear(
            torch::nn::LinearOptions(7, 128).bias(true)));
        torch::nn::init::orthogonal_(ego_linear1->weight, std::sqrt(2.0));
        torch::nn::init::constant_(ego_linear1->bias, 0.0);
        ego_linear2 = register_module("ego_linear2", torch::nn::Linear(
            torch::nn::LinearOptions(128, 128).bias(true)));
        torch::nn::init::orthogonal_(ego_linear2->weight, std::sqrt(2.0));
        torch::nn::init::constant_(ego_linear2->bias, 0.0);

        if (use_fused_kernel) {
            // Fused mode: single FC -> Max (no intermediate layer)
            // Road: 13 -> 128 (6 continuous + 7 one-hot)
            road_W = register_parameter("road_W", torch::empty({128, 13}));
            road_b = register_parameter("road_b", torch::zeros({128}));
            torch::nn::init::orthogonal_(road_W, std::sqrt(2.0));

            // Partner: 7 -> 128
            partner_W = register_parameter("partner_W", torch::empty({128, 7}));
            partner_b = register_parameter("partner_b", torch::zeros({128}));
            torch::nn::init::orthogonal_(partner_W, std::sqrt(2.0));
        } else {
            // Torch mode: Linear -> LayerNorm -> Linear -> Max
            // Road: 13 -> 128 -> 128
            road_linear1 = register_module("road_linear1", torch::nn::Linear(
                torch::nn::LinearOptions(13, 128).bias(true)));
            torch::nn::init::orthogonal_(road_linear1->weight, std::sqrt(2.0));
            torch::nn::init::constant_(road_linear1->bias, 0.0);
            road_ln = register_module("road_ln", torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({128})));
            road_linear2 = register_module("road_linear2", torch::nn::Linear(
                torch::nn::LinearOptions(128, 128).bias(true)));
            torch::nn::init::orthogonal_(road_linear2->weight, std::sqrt(2.0));
            torch::nn::init::constant_(road_linear2->bias, 0.0);

            // Partner: 7 -> 128 -> 128
            partner_linear1 = register_module("partner_linear1", torch::nn::Linear(
                torch::nn::LinearOptions(7, 128).bias(true)));
            torch::nn::init::orthogonal_(partner_linear1->weight, std::sqrt(2.0));
            torch::nn::init::constant_(partner_linear1->bias, 0.0);
            partner_ln = register_module("partner_ln", torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({128})));
            partner_linear2 = register_module("partner_linear2", torch::nn::Linear(
                torch::nn::LinearOptions(128, 128).bias(true)));
            torch::nn::init::orthogonal_(partner_linear2->weight, std::sqrt(2.0));
            torch::nn::init::constant_(partner_linear2->bias, 0.0);
        }

        // Shared embedding: 3*128 -> hidden
        shared_linear = register_module("shared_linear", torch::nn::Linear(
            torch::nn::LinearOptions(3*128, hidden).bias(true)));
        torch::nn::init::orthogonal_(shared_linear->weight, std::sqrt(2.0));
        torch::nn::init::constant_(shared_linear->bias, 0.0);
    }

    Tensor forward(Tensor x) override {
        int64_t B = x.size(0);
        auto target_dtype = ego_linear1->weight.dtype();
        x = x.to(target_dtype);

        // Split observations: ego (7), partner (441), road (1400)
        Tensor ego_obs = x.narrow(1, 0, 7);
        Tensor partner_obs = x.narrow(1, 7, 63*7);
        Tensor road_obs = x.narrow(1, 7 + 63*7, 200*7);

        // Ego encoding: Linear -> ReLU -> Linear (single point, no max)
        Tensor ego_features = ego_linear2->forward(torch::relu(ego_linear1->forward(ego_obs)));

        // Partner encoding
        Tensor partner_objects = partner_obs.view({B, 63, 7}).contiguous();
        Tensor partner_features;
        if (use_fused_kernel) {
            // Fused FC -> Max kernel
            partner_features = fc_max(partner_objects, partner_W, partner_b);
        } else {
            // Torch: Linear -> LayerNorm -> Linear -> Max
            auto h = partner_linear1->forward(partner_objects);  // (B, 63, 128)
            h = partner_ln->forward(h);
            h = partner_linear2->forward(h);  // (B, 63, 128)
            partner_features = std::get<0>(h.max(1));  // (B, 128)
        }

        // Road encoding with one-hot
        Tensor road_objects = road_obs.view({B, 200, 7});
        Tensor road_continuous = road_objects.narrow(2, 0, 6);
        Tensor road_categorical = road_objects.narrow(2, 6, 1).squeeze(2);
        Tensor road_onehot = torch::one_hot(road_categorical.to(torch::kInt64), 7).to(x.dtype());
        Tensor road_combined = torch::cat({road_continuous, road_onehot}, 2).contiguous();  // (B, 200, 13)

        Tensor road_features;
        if (use_fused_kernel) {
            // Fused FC -> Max kernel
            road_features = fc_max(road_combined, road_W, road_b);
        } else {
            // Torch: Linear -> LayerNorm -> Linear -> Max
            auto h = road_linear1->forward(road_combined);  // (B, 200, 128)
            h = road_ln->forward(h);
            h = road_linear2->forward(h);  // (B, 200, 128)
            road_features = std::get<0>(h.max(1));  // (B, 128)
        }

        // Concatenate and shared embedding: GELU -> Linear -> ReLU
        Tensor concat_features = torch::cat({ego_features, road_features, partner_features}, 1);
        Tensor embedding = torch::relu(shared_linear->forward(torch::gelu(concat_features)));
        return embedding;
    }
};

// G2048 decoder: separate policy and value heads, cat + narrow for contiguous output
class G2048Decoder : public Decoder {
    public:
        torch::nn::Linear dec_linear1{nullptr};
        torch::nn::Linear dec_linear2{nullptr};
        torch::nn::Linear val_linear1{nullptr};
        torch::nn::Linear val_linear2{nullptr};
        int hidden;
        int output;

    G2048Decoder(int hidden, int output)
        : hidden(hidden), output(output) {
        // Decoder head: hidden -> hidden -> num_atns
        dec_linear1 = register_module("dec_linear1", torch::nn::Linear(
            torch::nn::LinearOptions(hidden, hidden).bias(false)));
        torch::nn::init::orthogonal_(dec_linear1->weight, std::sqrt(2.0));

        dec_linear2 = register_module("dec_linear2", torch::nn::Linear(
            torch::nn::LinearOptions(hidden, output).bias(false)));
        torch::nn::init::orthogonal_(dec_linear2->weight, 0.01);

        // Value head: hidden -> hidden -> 1
        val_linear1 = register_module("val_linear1", torch::nn::Linear(
            torch::nn::LinearOptions(hidden, hidden).bias(false)));
        torch::nn::init::orthogonal_(val_linear1->weight, std::sqrt(2.0));

        val_linear2 = register_module("val_linear2", torch::nn::Linear(
            torch::nn::LinearOptions(hidden, 1).bias(false)));
        torch::nn::init::orthogonal_(val_linear2->weight, 1.0);
    }

    std::tuple<Logits, Tensor> forward(Tensor h) override {
        // Policy head
        Tensor logits = torch::gelu(dec_linear1->forward(h));
        logits = dec_linear2->forward(logits);

        // Value head
        Tensor value = torch::gelu(val_linear1->forward(h));
        value = val_linear2->forward(value);

        // Cat and narrow for contiguous outputs
        Tensor out = torch::cat({logits, value}, 1).contiguous();
        logits = out.narrow(1, 0, output);
        value = out.narrow(1, output, 1);

        return {Logits{logits, Tensor()}, value.squeeze(1)};
    }
};

// Create policy with env-specific encoder/decoder
Policy* create_policy(const std::string& env_name, int input_size, int hidden_size,
        int decoder_output_size, int num_layers, int act_n, bool is_continuous, bool kernels) {
    shared_ptr<Encoder> enc;
    shared_ptr<Decoder> dec;
    if (env_name == "puffer_snake") {
        enc = std::make_shared<SnakeEncoder>(input_size, hidden_size, 8);
        dec = std::make_shared<DefaultDecoder>(hidden_size, decoder_output_size, is_continuous);
    } else if (env_name == "falsepuffer_g2048") {
        enc = std::make_shared<SimpleG2048Encoder>(input_size, hidden_size);
        dec = std::make_shared<DefaultDecoder>(hidden_size, decoder_output_size, is_continuous);
    } else if (env_name == "puffer_nmmo3") {
        enc = std::make_shared<NMMO3Encoder>(input_size, hidden_size);
        dec = std::make_shared<NMMO3Decoder>(hidden_size, decoder_output_size);
    } else if (env_name == "puffer_drive") {
        enc = std::make_shared<DriveEncoder>(input_size, hidden_size);
        dec = std::make_shared<DefaultDecoder>(hidden_size, decoder_output_size, is_continuous);
    } else {
        enc = std::make_shared<DefaultEncoder>(input_size, hidden_size);
        dec = std::make_shared<DefaultDecoder>(hidden_size, decoder_output_size, is_continuous);
    }
    auto rnn = std::make_shared<MinGRU>(hidden_size, num_layers, kernels);
    return new Policy(enc, dec, rnn, input_size, act_n, hidden_size);
}
