#ifndef BAT_RECORD_H
#define BAT_RECORD_H

static inline void record_write_le16(FILE* f, unsigned int v) {
    fputc((int)(v & 0xffu), f);
    fputc((int)((v >> 8) & 0xffu), f);
}

static inline void record_write_le32(FILE* f, unsigned int v) {
    fputc((int)(v & 0xffu), f);
    fputc((int)((v >> 8) & 0xffu), f);
    fputc((int)((v >> 16) & 0xffu), f);
    fputc((int)((v >> 24) & 0xffu), f);
}

static inline void record_write_wav_header(FILE* f, int data_bytes) {
    int byte_rate = AUDIO_SAMPLE_RATE * 2;
    fwrite("RIFF", 1, 4, f);
    record_write_le32(f, 36u + (unsigned int)data_bytes);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    record_write_le32(f, 16);
    record_write_le16(f, 1);
    record_write_le16(f, 1);
    record_write_le32(f, AUDIO_SAMPLE_RATE);
    record_write_le32(f, (unsigned int)byte_rate);
    record_write_le16(f, 2);
    record_write_le16(f, 16);
    fwrite("data", 1, 4, f);
    record_write_le32(f, (unsigned int)data_bytes);
}

static inline void record_init(Bat* env, Client* client) {
    if (!env->record_video || client->recording_initialized) return;
    client->recording_initialized = 1;
    client->record_fps = env->record_video_fps;
    client->record_audio = env->record_video_audio ? 1 : 0;
    client->record_max_frames = client->record_fps * env->record_video_seconds;
    snprintf(client->record_frame_dir, sizeof(client->record_frame_dir),
        "recordings/bat_recording_frames");
    snprintf(client->record_wav_path, sizeof(client->record_wav_path),
        "recordings/bat_recording.wav");
    snprintf(client->record_mp4_path, sizeof(client->record_mp4_path),
        "recordings/bat_recording.mp4");
    system("mkdir -p recordings recordings/bat_recording_frames");
    if (client->record_audio) {
        client->record_wav = fopen(client->record_wav_path, "wb");
        if (client->record_wav != NULL) {
            record_write_wav_header(client->record_wav, 0);
        }
    }
    printf("Bat recording enabled: %s (%d fps, %d frames)\n",
        client->record_mp4_path, client->record_fps, client->record_max_frames);
}

static inline void record_enqueue_chirp(Bat* env) {
    Client* client = env->client;
    if (client == NULL || !client->recording_initialized ||
            client->recording_finalized || !client->record_audio) {
        return;
    }
    if (env->audio_chirp_serial <= 0 ||
            env->audio_chirp_serial == client->record_last_audio_chirp_serial) {
        return;
    }
    client->record_last_audio_chirp_serial = env->audio_chirp_serial;
    int voice_idx = client->record_voice_cursor;
    client->record_voice_cursor = (client->record_voice_cursor + 1) % RECORD_MAX_VOICES;
    BatRecordVoice* voice = &client->record_voices[voice_idx];
    voice->active = 1;
    voice->start_sample = client->record_audio_sample_cursor;
    voice->start_freq = env->last_chirp_start_freq;
    voice->end_freq = env->last_chirp_end_freq;
    voice->duration = chirp_audio_duration_at_fps(
        env->last_chirp_duration, client->record_fps);
}

static inline void record_append_audio_frame(Bat* env) {
    Client* client = env->client;
    if (client == NULL || !client->record_audio || client->record_wav == NULL) return;
    int frame_samples = AUDIO_SAMPLE_RATE / client->record_fps;
    for (int i = 0; i < frame_samples; i++) {
        int sample_index = client->record_audio_sample_cursor + i;
        float mixed = 0.0f;
        for (int v = 0; v < RECORD_MAX_VOICES; v++) {
            BatRecordVoice* voice = &client->record_voices[v];
            if (!voice->active) continue;
            int local_sample = sample_index - voice->start_sample;
            int voice_samples = (int)ceilf(voice->duration * AUDIO_SAMPLE_RATE);
            if (local_sample < 0) continue;
            if (local_sample >= voice_samples) {
                voice->active = 0;
                continue;
            }
            mixed += chirp_audio_sample_f32(voice->start_freq, voice->end_freq,
                voice->duration, local_sample, AUDIO_SAMPLE_RATE);
        }
        short pcm = (short)(bat_clampf(mixed, -1.0f, 1.0f) * 32767.0f);
        fwrite(&pcm, sizeof(short), 1, client->record_wav);
        client->record_audio_data_bytes += (int)sizeof(short);
    }
    client->record_audio_sample_cursor += frame_samples;
}

static inline void record_finalize(Client* client) {
    if (client == NULL || !client->recording_initialized ||
            client->recording_finalized) {
        return;
    }
    client->recording_finalized = 1;
    if (client->record_wav != NULL) {
        fseek(client->record_wav, 0, SEEK_SET);
        record_write_wav_header(client->record_wav, client->record_audio_data_bytes);
        fclose(client->record_wav);
        client->record_wav = NULL;
    }

    char cmd[1024];
    if (client->record_audio) {
        snprintf(cmd, sizeof(cmd),
            "ffmpeg -y -framerate %d -i %s/%%06d.png -i %s -frames:v %d "
            "-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest %s",
            client->record_fps, client->record_frame_dir, client->record_wav_path,
            client->record_frame, client->record_mp4_path);
    } else {
        snprintf(cmd, sizeof(cmd),
            "ffmpeg -y -framerate %d -i %s/%%06d.png -frames:v %d "
            "-c:v libx264 -pix_fmt yuv420p %s",
            client->record_fps, client->record_frame_dir, client->record_frame,
            client->record_mp4_path);
    }
    int status = system(cmd);
    if (status == 0) {
        printf("Bat recording saved: %s\n", client->record_mp4_path);
    } else {
        printf("Bat recording ffmpeg command failed with status %d\n", status);
    }
}

static inline void record_capture_frame(Bat* env) {
    Client* client = env->client;
    if (client == NULL || !client->recording_initialized ||
            client->recording_finalized) {
        return;
    }
    if (client->record_frame >= client->record_max_frames) {
        record_finalize(client);
        return;
    }
    record_enqueue_chirp(env);
    char path[512];
    snprintf(path, sizeof(path), "%s/%06d.png", client->record_frame_dir,
        client->record_frame);
    Image image = LoadImageFromScreen();
    ExportImage(image, path);
    UnloadImage(image);
    record_append_audio_frame(env);
    client->record_frame += 1;
    if (client->record_frame >= client->record_max_frames) {
        record_finalize(client);
    }
}

#endif
