#ifndef BAT_AUDIO_H
#define BAT_AUDIO_H

static inline float chirp_audio_duration_at_fps(float duration_norm, int fps) {
    float duration = chirp_duration_seconds(duration_norm);
    float scale = 60.0f / (float)fps;
    if (scale < 1.0f) scale = 1.0f;
    return duration * scale;
}

static inline float chirp_audio_duration_seconds(Bat* env, float duration_norm) {
    return chirp_audio_duration_at_fps(duration_norm, env->render_target_fps);
}

static inline float chirp_audio_frequency_hz(float freq_norm) {
    return AUDIO_MIN_HZ + freq_norm
        * (AUDIO_MAX_HZ - AUDIO_MIN_HZ);
}

static inline float chirp_audio_envelope(float t_norm) {
    if (t_norm <= 0.0f || t_norm >= 1.0f) return 0.0f;
    return bat_clampf(fminf(t_norm / AUDIO_ENVELOPE_FADE,
        (1.0f - t_norm) / AUDIO_ENVELOPE_FADE), 0.0f, 1.0f);
}

static inline float chirp_audio_sample_f32(float start_norm, float end_norm,
        float duration_seconds, int sample_index, int sample_rate) {
    float t = sample_index / (float)sample_rate;
    if (t >= duration_seconds) return 0.0f;

    float start_hz = chirp_audio_frequency_hz(start_norm);
    float end_hz = chirp_audio_frequency_hz(end_norm);
    float chirp_rate = (end_hz - start_hz) / duration_seconds;
    float phase = TWO_PI * (start_hz * t + 0.5f * chirp_rate * t * t);
    float envelope = chirp_audio_envelope(t / duration_seconds);
    return AUDIO_VOLUME * envelope * sinf(phase);
}

#ifndef BAT_HEADLESS
static inline void unload_chirp_sound(Client* client, int i) {
    if (!client->chirp_sound_loaded[i]) return;
    UnloadSound(client->chirp_sounds[i]);
    client->chirp_sound_loaded[i] = 0;
}

static inline void cleanup_audio(Client* client) {
    if (!client->audio_ready) return;
    for (int i = 0; i < AUDIO_VOICES; i++) {
        if (client->chirp_sound_loaded[i] && !IsSoundPlaying(client->chirp_sounds[i])) {
            unload_chirp_sound(client, i);
        }
    }
}

static inline void play_chirp_audio(Bat* env) {
    Client* client = env->client;
    if (client == NULL || !client->audio_ready) return;
    cleanup_audio(client);
    if (env->audio_chirp_serial <= 0 ||
            env->audio_chirp_serial == client->last_audio_chirp_serial) {
        return;
    }
    client->last_audio_chirp_serial = env->audio_chirp_serial;

    float duration = chirp_audio_duration_seconds(env, env->last_chirp_duration);
    int sample_count = (int)ceilf(duration * AUDIO_SAMPLE_RATE);

    short* samples = (short*)malloc(sample_count * sizeof(short));
    if (samples == NULL) return;
    for (int i = 0; i < sample_count; i++) {
        float sample = chirp_audio_sample_f32(env->last_chirp_start_freq,
            env->last_chirp_end_freq, duration, i, AUDIO_SAMPLE_RATE);
        samples[i] = (short)(bat_clampf(sample, -1.0f, 1.0f) * 32767.0f);
    }

    Wave wave = {
        .frameCount = (unsigned int)sample_count,
        .sampleRate = AUDIO_SAMPLE_RATE,
        .sampleSize = 16,
        .channels = 1,
        .data = samples,
    };
    Sound sound = LoadSoundFromWave(wave);
    UnloadWave(wave);

    int voice = client->audio_voice_cursor;
    client->audio_voice_cursor = (client->audio_voice_cursor + 1) % AUDIO_VOICES;
    unload_chirp_sound(client, voice);
    client->chirp_sounds[voice] = sound;
    client->chirp_sound_loaded[voice] = 1;
    SetSoundVolume(client->chirp_sounds[voice], 1.0f);
    PlaySound(client->chirp_sounds[voice]);
}
#endif

#endif
