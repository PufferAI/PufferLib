#include <float.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "ini.h"

#define PUF_TABLE_MAX_COLS 512

typedef struct {
    char name[64];
    int rows;
    int cols;
    char* labels[PUF_TABLE_MAX_COLS];
    float* values;
} Table;

static char* table_strdup(const char* s) {
    size_t n = strlen(s) + 1;
    char* out = (char*)malloc(n);
    if (!out) {
        perror("malloc");
        exit(1);
    }
    memcpy(out, s, n);
    return out;
}

static int table_col(Table* table, const char* label) {
    for (int i = 0; i < table->cols; i++) {
        if (strcmp(table->labels[i], label) == 0) {
            return i;
        }
    }
    return -1;
}

static int table_add_col(Table* table, const char* label) {
    if (table->cols >= PUF_TABLE_MAX_COLS) {
        fprintf(stderr, "table %s has too many columns\n", table->name);
        exit(1);
    }

    int col = table->cols++;
    table->labels[col] = table_strdup(label);
    float* old = table->values;
    table->values = (float*)calloc((size_t)table->rows * (size_t)table->cols, sizeof(float));
    if (table->rows > 0 && !table->values) {
        perror("realloc");
        exit(1);
    }
    for (int r = 0; r < table->rows; r++) {
        for (int c = 0; c < table->cols - 1; c++) {
            table->values[r * table->cols + c] = old[r * (table->cols - 1) + c];
        }
    }
    free(old);
    return col;
}

static int table_require_col(Table* table, const char* label) {
    int col = table_col(table, label);
    if (col >= 0) {
        return col;
    }
    fprintf(stderr, "table %s missing column %s\n", table->name, label);
    exit(1);
}

static int table_ensure_col(Table* table, const char* label) {
    int col = table_col(table, label);
    if (col >= 0) {
        return col;
    }
    return table_add_col(table, label);
}

static void table_resize_rows(Table* table, int rows) {
    if (rows == table->rows) {
        return;
    }

    float* old = table->values;
    int old_rows = table->rows;
    table->values = (float*)calloc((size_t)rows * (size_t)table->cols, sizeof(float));
    if (rows > 0 && table->cols > 0 && !table->values) {
        perror("calloc");
        exit(1);
    }
    for (int r = 0; r < old_rows && r < rows; r++) {
        for (int c = 0; c < table->cols; c++) {
            table->values[r * table->cols + c] = old[r * table->cols + c];
        }
    }
    free(old);
    table->rows = rows;
}

static void table_set(Table* table, int row, int col, float value) {
    table->values[row * table->cols + col] = value;
}

static float table_get(Table* table, int row, int col) {
    return table->values[row * table->cols + col];
}

static void table_free(Table* table) {
    for (int i = 0; i < table->cols; i++) {
        free(table->labels[i]);
    }
    free(table->values);
    memset(table, 0, sizeof(*table));
}

#ifdef PUFFER_CACHE_DATA

#include <dirent.h>
#include <sys/stat.h>

static const char* EXTRA_KEYS[] = {
    "train/learning_rate",
    "train/ent_coef",
    "train/gamma",
    "train/gae_lambda",
    "train/clip_coef",
    "train/vf_clip_coef",
    "train/vf_coef",
    "train/max_grad_norm",
    "train/momentum",
    "train/horizon",
    "train/replay_ratio",
    "train/minibatch_size",
    "policy/hidden_size",
    "policy/num_layers",
    "vec/total_agents",
    "train/total_timesteps",
};

static int has_suffix(const char* s, const char* suffix) {
    size_t n = strlen(s);
    size_t m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static int is_dir(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static void key_to_cache(char* out, size_t out_size, const char* section, const char* key) {
    if (strcmp(section, "config") == 0) {
        if (strncmp(key, "base.", 5) == 0) {
            snprintf(out, out_size, "%s", key + 5);
        } else {
            snprintf(out, out_size, "%s", key);
        }
    } else if (strcmp(section, "base") == 0) {
        snprintf(out, out_size, "%s", key);
    } else {
        snprintf(out, out_size, "%s/%s", section, key);
    }

    for (char* p = out; *p; p++) {
        if (*p == '.') {
            *p = '/';
        }
    }
}

static int parse_list(char* raw, float** out, int* len) {
    double* parsed = NULL;
    int n = 0;
    if (!puf_ini_parse_list(raw, &parsed, &n)) {
        return 0;
    }

    float* vals = (float*)calloc(n, sizeof(float));
    if (!vals) {
        perror("calloc");
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        vals[i] = (float)parsed[i];
    }

    free(parsed);
    *out = vals;
    *len = n;
    return 1;
}

static int load_ini_log(const char* path, Dict* scalars, Table* metrics) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        return 0;
    }

    char section[256] = "base";
    char* line = NULL;
    int cap = 0;
    while (puf_ini_read_line(fp, &line, &cap)) {
        puf_ini_strip_comment(line);
        char* s = puf_ini_trim(line);
        if (!*s) {
            continue;
        }

        size_t len = strlen(s);
        if (s[0] == '[' && len > 2 && s[len - 1] == ']') {
            s[len - 1] = 0;
            snprintf(section, sizeof(section), "%s", puf_ini_trim(s + 1));
            continue;
        }

        char* eq = strchr(s, '=');
        if (!eq) {
            free(line);
            fclose(fp);
            return 0;
        }
        *eq = 0;
        char* key = puf_ini_trim(s);
        char* val = puf_ini_trim(eq + 1);
        puf_ini_strip_quotes(val);

        if (strcmp(section, "metrics") == 0) {
            if (strstr(key, "loss")) {
                continue;
            }
            float* values = NULL;
            int n = 0;
            if (!parse_list(val, &values, &n)) {
                free(line);
                fclose(fp);
                return 0;
            }
            if (metrics->rows == 0) {
                table_resize_rows(metrics, n);
            } else if (metrics->rows != n) {
                free(values);
                free(line);
                fclose(fp);
                return 0;
            }
            int col = table_ensure_col(metrics, key);
            for (int r = 0; r < n; r++) {
                table_set(metrics, r, col, values[r]);
            }
            free(values);
        } else {
            double value = 0;
            if (!puf_ini_parse_val(val, &value)) {
                continue;
            }
            char full[512];
            key_to_cache(full, sizeof(full), section, key);
            dict_set(scalars, full, value);
        }
    }

    free(line);
    fclose(fp);
    return metrics->rows > 0;
}

static void table_copy_rows(Table* dst, int dst_row, Table* src) {
    for (int c = 0; c < src->cols; c++) {
        int out_col = table_ensure_col(dst, src->labels[c]);
        for (int r = 0; r < src->rows; r++) {
            table_set(dst, dst_row + r, out_col, table_get(src, r, c));
        }
    }
}

static void table_fill_scalar(Table* dst, int row, int rows, const char* key, float value) {
    int col = table_ensure_col(dst, key);
    for (int r = 0; r < rows; r++) {
        table_set(dst, row + r, col, value);
    }
}

static void load_env(const char* env, int full_dataset, Table* out) {
    char dir[1024];
    snprintf(dir, sizeof(dir), "logs/%s", env);
    struct dirent** ents = NULL;
    int nents = scandir(dir, &ents, NULL, alphasort);
    if (nents < 0) {
        return;
    }

    for (int i = 0; i < nents; i++) {
        if (ents[i]->d_name[0] == '.') {
            free(ents[i]);
            continue;
        }

        char path[2048];
        snprintf(path, sizeof(path), "%s/%s", dir, ents[i]->d_name);
        free(ents[i]);
        if (is_dir(path) || (!has_suffix(path, ".ini") && !has_suffix(path, ".log"))) {
            continue;
        }

        Dict scalars = {0};
        Table metrics = {0};
        if (!load_ini_log(path, &scalars, &metrics)) {
            dict_clear(&scalars);
            table_free(&metrics);
            continue;
        }

        int start = out->rows;
        table_resize_rows(out, out->rows + metrics.rows);
        table_copy_rows(out, start, &metrics);
        for (int s = 0; s < scalars.size; s++) {
            DictItem* item = &scalars.items[s];
            table_fill_scalar(out, start, metrics.rows, item->key, (float)item->value);
        }
        for (int k = 0; k < (int)(sizeof(EXTRA_KEYS) / sizeof(EXTRA_KEYS[0])); k++) {
            if (!dict_find(&scalars, EXTRA_KEYS[k])) {
                table_fill_scalar(out, start, metrics.rows, EXTRA_KEYS[k], 0);
            }
        }

        dict_clear(&scalars);
        table_free(&metrics);
    }
    free(ents);

    int steps_col = table_col(out, "agent_steps");
    int total_steps_col = table_col(out, "train/total_timesteps");
    for (int r = 0; r < out->rows; r++) {
        if (steps_col >= 0) {
            table_set(out, r, steps_col, table_get(out, r, steps_col) / 1e6f);
        }
        if (total_steps_col >= 0) {
            table_set(out, r, total_steps_col, table_get(out, r, total_steps_col) / 1e6f);
        }
    }

    if (full_dataset) {
        return;
    }

    int cost_col = table_col(out, "uptime");
    int score_col = table_col(out, "env/score");
    if (score_col < 0) {
        score_col = table_col(out, "env/perf");
    }
    if (cost_col < 0 || score_col < 0) {
        return;
    }

    int n = out->rows;
    unsigned char* keep = (unsigned char*)calloc((size_t)n, 1);
    for (int i = 0; i < n; i++) {
        keep[i] = 1;
        for (int j = 0; j < n; j++) {
            if (table_get(out, j, score_col) >= table_get(out, i, score_col) &&
                    table_get(out, j, cost_col) < table_get(out, i, cost_col)) {
                keep[i] = 0;
                break;
            }
        }
    }

    int w = 0;
    for (int r = 0; r < n; r++) {
        if (!keep[r]) {
            continue;
        }
        for (int c = 0; c < out->cols; c++) {
            table_set(out, w, c, table_get(out, r, c));
        }
        w++;
    }
    out->rows = w;
    free(keep);
}

static void write_env(FILE* fp, const char* env, Table* table) {
    if (table->rows == 0) {
        return;
    }

    fprintf(fp, "\n[%s]\n", env);
    for (int c = 0; c < table->cols; c++) {
        fprintf(fp, "%s = ", table->labels[c]);
        for (int r = 0; r < table->rows; r++) {
            if (r > 0) {
                fputc(',', fp);
            }
            fprintf(fp, "%.6g", table_get(table, r, c));
        }
        fputc('\n', fp);
    }
}

int main(int argc, char** argv) {
    int full_dataset = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--full") == 0) {
            full_dataset = 1;
        }
    }

    if (is_dir("resources/constellation") == 0) {
        mkdir("resources/constellation", 0777);
    }

    FILE* fp = fopen("resources/constellation/experiments.ini", "w");
    if (!fp) {
        perror("resources/constellation/experiments.ini");
        return 1;
    }
    fprintf(fp, "# PufferLib constellation cache v1\n");

    struct dirent** ents = NULL;
    int nents = scandir("logs", &ents, NULL, alphasort);
    if (nents > 0) {
        for (int i = 0; i < nents; i++) {
            if (ents[i]->d_name[0] == '.') {
                free(ents[i]);
                continue;
            }

            char path[1024];
            snprintf(path, sizeof(path), "logs/%s", ents[i]->d_name);
            if (is_dir(path)) {
                Table table = {0};
                snprintf(table.name, sizeof(table.name), "%s", ents[i]->d_name);
                load_env(ents[i]->d_name, full_dataset, &table);
                write_env(fp, ents[i]->d_name, &table);
                table_free(&table);
            }
            free(ents[i]);
        }
        free(ents);
    }

    fclose(fp);
    return 0;
}

#else

#include "raylib.h"

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#include "rcamera.h"

#if defined(PLATFORM_DESKTOP) || defined(PLATFORM_DESKTOP_SDL)
    #if defined(GRAPHICS_API_OPENGL_ES2)
        #include "glad_gles2.h"       // Required for: OpenGL functionality
        #define glGenVertexArrays glGenVertexArraysOES
        #define glBindVertexArray glBindVertexArrayOES
        #define glDeleteVertexArrays glDeleteVertexArraysOES
        #define GLSL_VERSION            100
    #else
        #if defined(__APPLE__)
            #define GL_SILENCE_DEPRECATION // Silence Opengl API deprecation warnings
            #include <OpenGL/gl3.h>     // OpenGL 3 library for OSX
            #include <OpenGL/gl3ext.h>  // OpenGL 3 extensions library for OSX
        #else
            #include "glad.h"       // Required for: OpenGL functionality
        #endif
        #define GLSL_VERSION            330
    #endif
#else   // PLATFORM_ANDROID, PLATFORM_WEB
    #include <GLES3/gl3.h>
    #define GLSL_VERSION            100
#endif

#include "rlgl.h"
#include "raymath.h"

#define CAMERA_ORBITAL_SPEED 0.05f
#define MOUSE_ROTATE_SPEED 0.005f
#define CAMERA_ZOOM_SPEED 0.08f
void CustomUpdateCamera(Camera *camera, Rectangle bounds) {
    Vector2 mouse = GetMousePosition();
    if (CheckCollisionPointRec(mouse, bounds) && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        CameraYaw(camera, -delta.x*MOUSE_ROTATE_SPEED, true);
        CameraPitch(camera, -delta.y*MOUSE_ROTATE_SPEED, true, true, false);
    } else {
        float dt = CAMERA_ORBITAL_SPEED*GetFrameTime();
        Matrix rotation = MatrixRotate(GetCameraUp(camera), dt);
        Vector3 view = Vector3Subtract(camera->position, camera->target);
        view = Vector3Transform(view, rotation);
        camera->position = Vector3Add(camera->target, view);
    }
    if (CheckCollisionPointRec(mouse, bounds)) {
        float dist = Vector3Distance(camera->position, camera->target);
        CameraMoveToTarget(camera, -GetMouseWheelMove() * dist * CAMERA_ZOOM_SPEED);
    }
    if (IsKeyPressed(KEY_KP_SUBTRACT)) CameraMoveToTarget(camera, 2.0f);
    if (IsKeyPressed(KEY_KP_ADD)) CameraMoveToTarget(camera, -2.0f);
}

#define SETTINGS_HEIGHT 20
#define SEP 8
#define SPACER 25
#define TOGGLE_WIDTH 70
#define DROPDOWN_WIDTH 125

#define LINEAR 0
#define LOG 1
#define LOGIT 2

#define PUFF_CYAN ((Color){0, 187, 187, 255})
#define PUFF_WHITE ((Color){241, 241, 241, 255})
#define PUFF_BACKGROUND ((Color){6, 24, 24, 255})

int hyper_count = 21;
char *hyper_key[21] = {
    "uptime",
    "agent_steps",
    "parameters",
    "tflops",
    "env/perf",
    "env/score",
    "train/learning_rate",
    "train/ent_coef",
    "train/gamma",
    "train/gae_lambda",
    "train/clip_coef",
    "train/vf_clip_coef",
    "train/vf_coef",
    "train/max_grad_norm",
    "train/momentum",
    "train/horizon",
    "train/replay_ratio",
    "train/minibatch_size",
    "policy/hidden_size",
    "policy/num_layers",
    "vec/total_agents",
};
char *hyper_label[21] = {
    "uptime",
    "agent_steps_M",
    "params_M",
    "flops_T",
    "perf",
    "score",
    "learning_rate",
    "ent_coef",
    "gamma",
    "gae_lambda",
    "clip_coef",
    "vf_clip_coef",
    "vf_coef",
    "max_grad_norm",
    "momentum",
    "horizon",
    "replay_ratio",
    "minibatch_size",
    "hidden_size",
    "num_layers",
    "total_agents",
};

typedef struct Glyph {
    float x;
    float y;
    float i;
    float r;
    float g;
    float b;
    float a;
} Glyph;

typedef struct Point {
    float x;
    float y;
    float z;
    float c;
} Point;

typedef struct {
    float click_x;
    float click_y;
    float x;
    float y;
    int env_idx;
    int ary_idx;
    bool active;
} Tooltip;

typedef struct {
    Table *tables;
    int n;
} Dataset;

typedef struct PlotArgs {
    float mmin[4];
    float mmax[4];
    int scale[4];
    int width;
    int height;
    int title_font_size;
    int axis_font_size;
    int axis_tick_font_size;
    int legend_font_size;
    int line_width;
    int tick_length;
    int top_margin;
    int bottom_margin;
    int left_margin;
    int right_margin;
    int tick_margin;
    Color font_color;
    Color background_color;
    Color axis_color;
    char* x_label;
    char* y_label;
    char* z_label;
    Font font;
    Font font_small;
    Camera3D camera;
} PlotArgs;

PlotArgs DEFAULT_PLOT_ARGS = {
    .mmin = {0.0f, 0.0f, 0.0f, 0.0f},
    .mmax = {0.0f, 0.0f, 0.0f, 0.0f},
    .scale = {0, 0, 0, 0},
    .width = 960,
    .height = 540 - SETTINGS_HEIGHT,
    .title_font_size = 32,
    .axis_font_size = 32,
    .axis_tick_font_size = 16,
    .legend_font_size = 12,
    .line_width = 2,
    .tick_length = 8,
    .tick_margin = 8,
    .top_margin = 70,
    .bottom_margin = 70,
    .left_margin = 100,
    .right_margin = 100,
    .font_color = PUFF_WHITE,
    .background_color = PUFF_BACKGROUND,
    .axis_color = PUFF_WHITE,
    .x_label = "Cost",
    .y_label = "Score",
    .z_label = "Train/Learning Rate",
};


Table* dataset_table(Dataset *data, char *env) {
    for (int i = 0; i < data->n; i++) {
        if (strcmp(data->tables[i].name, env) == 0) {
            return &data->tables[i];
        }
    }
    printf("Error: env %s not found\n", env);
    exit(1);
    return NULL;
}

float safe_log10(float x) {
    if (x <= 0) {
        return x;
    }
    return log10(x);
}

float scale_val(int scale, float val) {
    if (scale == LINEAR) {
        return val;
    } else if (scale == LOG) {
        return safe_log10(val);
    } else if (scale == LOGIT) {
        return safe_log10(1 - val);
    } else {
        return val;
    }
}

float unscale_val(int scale, float val) {
    if (scale == LINEAR) {
        return val;
    } else if (scale == LOG) {
        return powf(10, val);
    } else if (scale == LOGIT) {
        return 1 / (1 + powf(10, val));
    }
    return val;
}

Color rgb(float h) {
    return ColorFromHSV(120*(1.0 + h), 0.8f, 0.15f);
}

void draw_axes(PlotArgs args) {
    DrawLine(args.left_margin, args.top_margin,
        args.left_margin, args.height - args.bottom_margin, PUFF_WHITE);
    DrawLine(args.left_margin, args.height - args.bottom_margin,
        args.width - args.right_margin, args.height - args.bottom_margin, PUFF_WHITE);
}

const char* format_tick_label(double value) {
    static char buffer[32];

    if (fabs(value) < 1e-10) {
        strcpy(buffer, "0");
        return buffer;
    }

    if (fabs(value) < 0.001 || fabs(value) > 10000) {
        snprintf(buffer, sizeof(buffer), "%.3e", value);
    } else {
        snprintf(buffer, sizeof(buffer), "%.3f", value);
    }

    return buffer;
}

void label_ticks(char ticks[][32], PlotArgs args, int axis_idx, int tick_n) {
    float mmin = scale_val(args.scale[axis_idx], args.mmin[axis_idx]);
    float mmax = scale_val(args.scale[axis_idx], args.mmax[axis_idx]);
    for (int i=0; i<tick_n; i++) {
        float val = mmin + i*(mmax - mmin)/(tick_n - 1.0f);
        val = unscale_val(args.scale[axis_idx], val);
        const char* label = format_tick_label(val);
        strcpy(ticks[i], label);
    }
}

void draw_ticks(char x_ticks[][32], int x_n, char y_ticks[][32], int y_n, PlotArgs args) {
    int width = args.width;
    int height = args.height;

    float plot_width = width - args.left_margin - args.right_margin;
    float plot_height = height - args.top_margin - args.bottom_margin;

    for (int i=0; i<x_n; i++) {
        char* label = x_ticks[i];
        float x_pos = args.left_margin + i*plot_width/(x_n - 1.0f);
        float y_pos = args.height - args.bottom_margin;
        DrawLine(x_pos, y_pos - args.tick_length,
            x_pos, y_pos + args.tick_length, args.axis_color);
        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_tick_font_size, 0);
        DrawTextEx(args.font_small, label,
            (Vector2){
                x_pos - this_tick_size.x/2,
                y_pos + args.tick_length + args.tick_margin,
            },
            args.axis_tick_font_size, 0, PUFF_WHITE
        );
    }
    for (int i=0; i<y_n; i++) {
        float y_pos = height - args.bottom_margin - i*plot_height/(y_n - 1.0f);
        char* label = y_ticks[i];
        DrawLine(args.left_margin - args.tick_length, y_pos,
            args.left_margin + args.tick_length, y_pos, args.axis_color);
        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_tick_font_size, 0);
        DrawTextEx(args.font_small, label,
            (Vector2){
                args.left_margin - this_tick_size.x - args.tick_length - args.tick_margin,
                y_pos - this_tick_size.y/2,
            },
            args.axis_tick_font_size, 0, PUFF_WHITE
        );
    }
}

Vector2 compute_ticks(PlotArgs args) {
    int width = args.width;
    int height = args.height;

    float plot_width = width - args.left_margin - args.right_margin;
    float plot_height = height - args.top_margin - args.bottom_margin;

    Vector2 tick_label_size = MeasureTextEx(args.font, "estimate", args.axis_font_size, 0);
    int num_x_ticks = 1 + plot_width/tick_label_size.x;
    int num_y_ticks = 1 + plot_height/tick_label_size.y;

    return (Vector2){num_x_ticks, num_y_ticks};
}

void draw_all_ticks(PlotArgs args) {
    Vector2 tick_n = compute_ticks(args);
    char x_ticks[(int)tick_n.x][32];
    char y_ticks[(int)tick_n.y][32];
    label_ticks(x_ticks, args, 0, tick_n.x);
    label_ticks(y_ticks, args, 1, tick_n.y);
    draw_ticks(x_ticks, tick_n.x, y_ticks, tick_n.y, args);
}

void draw_box_ticks(char* hypers[], int n, PlotArgs args) {
    Vector2 tick_n = compute_ticks(args);
    char x_ticks[(int)tick_n.x][32];
    label_ticks(x_ticks, args, 0, tick_n.x);
    draw_ticks(x_ticks, tick_n.x, x_ticks, 0, args);
    float dy = (args.height - args.top_margin - args.bottom_margin) / (float)n;
    for (int i=0; i<n; i++) {
        float y_pos = args.top_margin + (i + 0.5f)*dy;
        DrawLine(args.left_margin - args.tick_length, y_pos,
            args.left_margin + args.tick_length, y_pos, args.axis_color);
        Vector2 ts = MeasureTextEx(args.font, hypers[i], args.axis_tick_font_size, 0);
        DrawTextEx(args.font_small, hypers[i],
            (Vector2){
                args.left_margin - ts.x - args.tick_length - args.tick_margin,
                y_pos - ts.y/2,
            },
            args.axis_tick_font_size, 0, PUFF_WHITE
        );
    }
}

void draw_axes3() {
    DrawLine3D(
        (Vector3){0, 0, 0},
        (Vector3){1, 0, 0},
        RED
    );
    DrawLine3D(
        (Vector3){0, 0, 0},
        (Vector3){0, 1, 0},
        GREEN
    );
    DrawLine3D(
        (Vector3){0, 0, 0},
        (Vector3){0, 0, 1},
        BLUE
    );
}

void plot_gl(Glyph* glyphs, int size, Shader* shader) {
    int n = size;

    GLuint vao = 0;
    GLuint vbo = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, n*sizeof(Glyph), glyphs, GL_STATIC_DRAW);
        glVertexAttribPointer(shader->locs[SHADER_LOC_VERTEX_POSITION], 3, GL_FLOAT, GL_FALSE, sizeof(Glyph), 0);
        glEnableVertexAttribArray(shader->locs[SHADER_LOC_VERTEX_POSITION]);
        int vertexColorLoc = shader->locs[SHADER_LOC_VERTEX_COLOR];
        glVertexAttribPointer(vertexColorLoc, 4, GL_FLOAT, GL_FALSE, sizeof(Glyph), (void*)(3*sizeof(float)));
        glEnableVertexAttribArray(vertexColorLoc);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    rlDrawRenderBatchActive();
    rlSetBlendFactors(GL_ONE, GL_ONE, GL_MAX);
    rlSetBlendMode(RL_BLEND_CUSTOM);
    int currentTimeLoc = GetShaderLocation(*shader, "currentTime");
    glUseProgram(shader->id);
        glUniform1f(currentTimeLoc, GetTime());
        Matrix modelViewProjection = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection());
        glUniformMatrix4fv(shader->locs[SHADER_LOC_MATRIX_MVP], 1, false, MatrixToFloat(modelViewProjection));
        glBindVertexArray(vao);
            glDrawArrays(GL_POINTS, 0, n);
        glBindVertexArray(0);
    glUseProgram(0);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    rlSetBlendMode(RL_BLEND_ALPHA);
}

void GuiDropdownFilter(int x, int y, char* options, int *selection, bool *dropdown_active,
        Vector2 focus, char *text1, float *text1_val, char *text2, float *text2_val) {
    Rectangle rect = {x, y, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
    if (GuiDropdownBox(rect, options, selection, *dropdown_active)) {
        *dropdown_active = !*dropdown_active;
    }
    Rectangle text1_rect = {x + DROPDOWN_WIDTH, y, TOGGLE_WIDTH, SETTINGS_HEIGHT};
    bool text1_active = CheckCollisionPointRec(focus, text1_rect);
    if (GuiTextBox(text1_rect, text1, 32, text1_active)) {
        *text1_val = atof(text1);
    }
    Rectangle text2_rect = {x + DROPDOWN_WIDTH + TOGGLE_WIDTH, y, TOGGLE_WIDTH, SETTINGS_HEIGHT};
    bool text2_active = CheckCollisionPointRec(focus, text2_rect);
    if (GuiTextBox(text2_rect, text2, 32, text2_active)) {
        *text2_val = atof(text2);
    }
}

void apply_filter(bool* filter, Table* table, int col, float min, float max) {
    for (int i=0; i<table->rows; i++) {
        float val = table_get(table, i, col);
        if (val < min || val > max) {
            filter[i] = false;
        }
    }
}

void autoscale(Point* points, int size, PlotArgs *args) {
    float mmin[4] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    float mmax[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (int i=0; i<size; i++) {
        float* vals = (float*)&points[i];
        for (int j=0; j<4; j++) {
            float val = vals[j];
            if (val < mmin[j]) mmin[j] = val;
            if (val > mmax[j]) mmax[j] = val;
        }
    }
    for (int j=0; j<4; j++) {
        args->mmin[j] = mmin[j];
        args->mmax[j] = mmax[j];
    }
}

void toPx(Point *points, Glyph* glyphs, int size, PlotArgs args) {
    float mmin[4];
    float mmax[4];
    float delta[4];
    for (int j=0; j<4; j++) {
        mmin[j] = scale_val(args.scale[j], args.mmin[j]);
        mmax[j] = scale_val(args.scale[j], args.mmax[j]);
        delta[j] = mmax[j] - mmin[j];
    }

    for (int i = 0; i < size; i++) {
        Point p = points[i];
        float xi = scale_val(args.scale[0], p.x);
        float yi = scale_val(args.scale[1], p.y);
        float zi = scale_val(args.scale[2], p.z);
        float px, py;

        if (args.mmin[2] != 0 || args.mmax[2] != 0) {
            Vector3 v = (Vector3){
                (xi - mmin[0])/delta[0],
                (yi - mmin[1])/delta[1],
                (zi - mmin[2])/delta[2]
            };
            assert(args.camera.fovy != 0);
            Vector2 screen_pos = GetWorldToScreenEx(v, args.camera, args.width, args.height);
            px = screen_pos.x;
            py = screen_pos.y;
        } else {
            // TODO: Check margins
            px = args.left_margin + (xi - mmin[0]) / delta[0] * (args.width - args.left_margin - args.right_margin);
            py = args.height - args.bottom_margin - (yi - mmin[1]) / delta[1] * (args.height - args.top_margin - args.bottom_margin);
        }

        float cmap = points[i].c;
        cmap = scale_val(args.scale[3], cmap);
        float c_min = mmin[3];
        float c_max = mmax[3];
        if (c_min != c_max) {
            cmap = (cmap - c_min)/(c_max - c_min);
        }
        Color c = rgb(cmap);
        glyphs[i] = (Glyph){
            px,
            py,
            i,
            c.r/255.0f,
            c.g/255.0f,
            c.b/255.0f,
            c.a/255.0f,
        };
    }
}

void update_closest(Tooltip* tooltip, Vector2 *indices, Glyph* glyphs, int size, float x_offset, float y_offset) {
    float dx = tooltip->click_x - tooltip->x;
    float dy = tooltip->click_y - tooltip->y;
    float dist = sqrt(dx*dx + dy*dy);

    for (int i=0; i<size; i++) {
        dx = x_offset + glyphs[i].x - tooltip->click_x;
        dy = y_offset + glyphs[i].y - tooltip->click_y;
        float d = sqrt(dx*dx + dy*dy);
        if (d < dist) {
            dist = d;
            tooltip->x = x_offset + glyphs[i].x;
            tooltip->y = y_offset + glyphs[i].y;
            tooltip->env_idx = indices[i].x;
            tooltip->ary_idx = indices[i].y;
        }
    }
}

void brighten_hit(Tooltip* tooltip, Vector2* indices, Glyph* glyphs, int size,
        float x_offset, float y_offset, bool follow) {
    if (!tooltip->active) {
        return;
    }
    for (int i=0; i<size; i++) {
        if (indices[i].y != tooltip->ary_idx || indices[i].x != tooltip->env_idx) {
            continue;
        }
        glyphs[i].r = fmaxf(glyphs[i].r * 1.25f, glyphs[i].r + 0.25f);
        glyphs[i].g = fmaxf(glyphs[i].g * 1.25f, glyphs[i].g + 0.25f);
        glyphs[i].b = fmaxf(glyphs[i].b * 1.25f, glyphs[i].b + 0.25f);
        glyphs[i].a = 1.5625f;
        if (follow) {
            tooltip->x = x_offset + glyphs[i].x;
            tooltip->y = y_offset + glyphs[i].y;
            follow = false;
        }
    }
}

void copy_hypers_to_clipboard(Table *table, char* buffer, int row) {
    char* start = buffer;
    char* prefix = NULL;
    int prefix_len = 0;
    for (int col = 0; col < table->cols; col++) {
        char *key = table->labels[col];
        char *slash = strchr(key, '/');
        if (!slash || row >= table->rows) {
            continue;
        }

        if (prefix == NULL || strncmp(prefix, key, prefix_len) != 0) {
            if (prefix != NULL) {
                buffer += sprintf(buffer, "\n");
            }
            prefix = key;
            prefix_len = slash - prefix;
            buffer += sprintf(buffer, "[");
            snprintf(buffer, prefix_len+1, "%s", prefix);
            buffer += prefix_len;
            buffer += sprintf(buffer, "]\n");
        }

        char* suffix = slash + 1;
        double val = table_get(table, row, col);
        if (strcmp(suffix, "total_timesteps") == 0) {
            // Use agent_steps (training-only) instead of total_timesteps (train+eval)
            int agent_steps = table_require_col(table, "agent_steps");
            val = table_get(table, row, agent_steps);
            buffer += sprintf(buffer, "%s = %lld\n", suffix, (long long)(val * 1e6));
        } else if (strcmp(suffix, "agent_steps") == 0) {
            buffer += sprintf(buffer, "%s = %lld\n", suffix, (long long)(val * 1e6));
        } else if (val == (long long)val) {
            buffer += sprintf(buffer, "%s = %lld\n", suffix, (long long)val);
        } else {
            buffer += sprintf(buffer, "%s = %g\n", suffix, val);
        }
    }
    buffer[0] = '\0';
    SetClipboardText(start);
}

Table* dataset_get_table(Dataset* data, const char* key) {
    for (int i = 0; i < data->n; i++) {
        if (strcmp(data->tables[i].name, key) == 0) {
            return &data->tables[i];
        }
    }
    data->tables = realloc(data->tables, (data->n + 1) * sizeof(Table));
    if (!data->tables) {
        perror("realloc");
        exit(1);
    }
    Table* table = &data->tables[data->n++];
    memset(table, 0, sizeof(*table));
    snprintf(table->name, sizeof(table->name), "%s", key);
    return table;
}

void table_add_values(Table* table, const char* key, const char* values) {
    double* vals = NULL;
    int len = 0;
    if (!puf_ini_parse_list(values, &vals, &len)) {
        fprintf(stderr, "constellation error: invalid values for %s\n", key);
        exit(1);
    }

    if (table->rows == 0) {
        table_resize_rows(table, len);
    }
    if (table->rows != len) {
        fprintf(stderr, "constellation error: column %s has %d values, expected %d\n",
            key, len, table->rows);
        exit(1);
    }

    int col = table_add_col(table, key);
    for (int i = 0; i < len; i++) {
        table_set(table, i, col, (float)vals[i]);
    }
    free(vals);
}

Dataset load_dataset(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "could not open %s\n", path);
        exit(1);
    }

    Dataset data = {0};
    char env_name[256] = "";
    char* line = NULL;
    int cap = 0;
    for (int n = 1; puf_ini_read_line(fp, &line, &cap); n++) {
        puf_ini_strip_comment(line);
        char* s = puf_ini_trim(line);
        if (!*s) {
            continue;
        }

        size_t len = strlen(s);
        if (s[0] == '[' && len >= 3 && s[len - 1] == ']') {
            s[len - 1] = 0;
            snprintf(env_name, sizeof(env_name), "%s", puf_ini_trim(s + 1));
            continue;
        }

        char* eq = strchr(s, '=');
        if (!eq) {
            fprintf(stderr, "%s:%d: expected key=value\n", path, n);
            exit(1);
        }
        if (!env_name[0]) {
            fprintf(stderr, "%s:%d: expected section before key=value\n", path, n);
            exit(1);
        }
        *eq = 0;
        char* key = puf_ini_trim(s);
        char* val = puf_ini_trim(eq + 1);
        puf_ini_strip_quotes(val);
        table_add_values(dataset_get_table(&data, env_name), key, val);
    }
    free(line);
    fclose(fp);
    return data;
}

int main(void) {
    Dataset data = load_dataset("resources/constellation/experiments.ini");
    for (int i = 0; i < data.n; i++) {
        Table* table = &data.tables[i];
        int hid = table_col(table, "policy/hidden_size");
        int layers = table_col(table, "policy/num_layers");
        int replay = table_col(table, "train/replay_ratio");
        int steps = table_col(table, "agent_steps");
        int pcol = table_ensure_col(table, "parameters");
        int fcol = table_ensure_col(table, "tflops");
        if (hid < 0 || layers < 0) {
            continue;
        }
        for (int r = 0; r < table->rows; r++) {
            float h = table_get(table, r, hid);
            float L = table_get(table, r, layers);
            float params = 3.0f * h * h * L;
            table_set(table, r, pcol, params * 1e-6f);
            if (replay < 0 || steps < 0 || params <= 0) {
                continue;
            }
            table_set(table, r, fcol, 6.0f * params
                * table_get(table, r, replay) * table_get(table, r, steps) * 1e-6f);
        }
    }
    int max_data_points = 0;
    for (int i=0; i<data.n; i++) {
        max_data_points = data.tables[i].rows > max_data_points ?
            data.tables[i].rows : max_data_points;
    }
    int total_points = 0;
    for (int i=0; i<data.n; i++) {
        total_points += data.tables[i].rows;
    }

    // Create options as a semicolon-separated string
    size_t options_len = 0;
    for (int i = 0; i < hyper_count; i++) {
        options_len += strlen(hyper_label[i]) + 1;
    }
    char *options = malloc(options_len);
    options[0] = '\0';
    for (int i = 0; i < hyper_count; i++) {
        if (i > 0) strcat(options, ";");
        strcat(options, hyper_label[i]);
    }

    // Options with extra "env_name;"
    char* extra = "env_name;";
    char *env_hyper_options = malloc(options_len + strlen(extra));
    strcpy(env_hyper_options, extra);
    strcat(env_hyper_options, options);

    char* z_extra = "none;";
    char *z_options = malloc(options_len + strlen(z_extra));
    strcpy(z_options, z_extra);
    strcat(z_options, options);

    // Env names as semi-colon-separated string
    size_t env_options_len = 4;
    for (int i = 0; i < data.n; i++) {
        env_options_len += strlen(data.tables[i].name) + 1;
    }
    char *env_options = malloc(env_options_len);
    strcpy(env_options, "all;");
    env_options[4] = '\0';
    for (int i = 0; i < data.n; i++) {
        if (i > 0) strcat(env_options, ";");
        strcat(env_options, data.tables[i].name);
    }

    char* clipboard = malloc(16384);

    // Points
    printf("total points: %d", total_points);
    Point* points = calloc(total_points, sizeof(Point));
    int glyph_cap = total_points * hyper_count;
    Glyph* glyphs = calloc(glyph_cap, sizeof(Glyph));
    Vector2* env_indices = calloc(glyph_cap, sizeof(Vector2));

    // Initialize Raylib
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(2*DEFAULT_PLOT_ARGS.width, DEFAULT_PLOT_ARGS.height + 2*SETTINGS_HEIGHT, "Puffer Constellation");
    Texture2D puffer = LoadTexture("resources/shared/puffers.png");

    DEFAULT_PLOT_ARGS.font = LoadFontEx("resources/shared/JetBrainsMono-SemiBold.ttf", 32, NULL, 255);
    DEFAULT_PLOT_ARGS.font_small = LoadFontEx("resources/shared/JetBrainsMono-SemiBold.ttf", 16, NULL, 255);
    Font gui_font = LoadFontEx("resources/shared/JetBrainsMono-SemiBold.ttf", 14, NULL, 255);

    GuiLoadStyle("resources/constellation/puffer.rgs");
    GuiSetFont(gui_font);
    ClearBackground(PUFF_BACKGROUND);
    SetTargetFPS(60);

    Shader shader = LoadShader(
        TextFormat("resources/constellation/point_particle_%i.vs", GLSL_VERSION),
        TextFormat("resources/constellation/point_particle_%i.fs", GLSL_VERSION)
    );
    Shader blur_shader = LoadShader(
        TextFormat("resources/constellation/blur_%i.vs", GLSL_VERSION),
        TextFormat("resources/constellation/blur_%i.fs", GLSL_VERSION)
    );

    // Allows the vertex shader to set the point size of each particle individually
    #ifndef GRAPHICS_API_OPENGL_ES2
    glEnable(GL_PROGRAM_POINT_SIZE);
    #endif

    PlotArgs args1 = DEFAULT_PLOT_ARGS;
    args1.camera = (Camera3D){ 0 };
    args1.camera.position = (Vector3){ 1.5f, 1.25f, 1.5f };
    args1.camera.target = (Vector3){ 0.5f, 0.5f, 0.5f };
    args1.camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    args1.camera.fovy = 45.0f;
    args1.camera.projection = CAMERA_PERSPECTIVE;
    args1.scale[0] = 1;
    args1.scale[2] = 1;
    args1.top_margin = 20;
    args1.left_margin = 100;
    args1.right_margin = 50;
    args1.bottom_margin = 50;
    RenderTexture2D fig1 = LoadRenderTexture(args1.width, args1.height);
    RenderTexture2D fig1_overlay = LoadRenderTexture(args1.width, args1.height);
    Rectangle fig1_bounds = {0, 2*SETTINGS_HEIGHT, args1.width, args1.height};
    int fig_env_idx = 0;
    bool fig_env_active = false;
    bool fig_x_active = false;
    int fig_x_idx = 0;
    bool fig_xscale_active = false;
    bool fig_y_active = false;
    int fig_y_idx = 4;
    bool fig_yscale_active = false;
    bool fig_z_active = false;
    int fig_z_idx = 0;
    bool fig_zscale_active = false;
    int fig_color_idx = 0;
    bool fig_color_active = false;
    bool fig_colorscale_active = false;
    bool fig_range1_active = false;
    int fig_range1_idx = 4;
    char fig_range1_min[32] = {0};
    char fig_range1_max[32] = {0};
    float fig_range1_min_val = 0;
    float fig_range1_max_val = FLT_MAX;
    bool fig_range2_active = false;
    int fig_range2_idx = 0;
    char fig_range2_min[32] = {0};
    char fig_range2_max[32] = {0};
    float fig_range2_min_val = FLT_MIN;
    float fig_range2_max_val = FLT_MAX;
    int fig_box_idx = LOG;
    bool fig_box_active = false;

    char* scale_options = "linear;log;logit";

    PlotArgs args2 = DEFAULT_PLOT_ARGS;
    RenderTexture2D fig2 = LoadRenderTexture(args2.width, args2.height);
    args2.x_label = "Value";
    args2.y_label = "Hyperparameter";
    args2.left_margin = 170;
    args2.right_margin = 50;
    args2.top_margin = 10;
    args2.bottom_margin = 50;
    Rectangle fig2_bounds = {args1.width, 2*SETTINGS_HEIGHT, args2.width, args2.height};

    int x, y, z, c;

    bool *filter = calloc(max_data_points, sizeof(bool));

    Tooltip tooltip = {0};

    Vector2 focus = {0, 0};

    while (!WindowShouldClose()) {
        bool right_clicked = false;

        BeginDrawing();
        ClearBackground(PUFF_BACKGROUND);

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            focus = GetMousePosition();
            tooltip.active = false;
        }
        if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON)) {
            Vector2 mouse_pos = GetMousePosition();
            right_clicked = true;
            tooltip.active = true;
            tooltip.click_x = mouse_pos.x;
            tooltip.click_y = mouse_pos.y;
        }

        int use_3d = fig_z_idx != 0;
        args1.x_label = hyper_label[fig_x_idx];
        args1.y_label = hyper_label[fig_y_idx];
        args1.z_label = use_3d ? hyper_label[fig_z_idx - 1] : "";
        int start = 0;
        int end = data.n;
        if (fig_env_idx != 0) {
            start = fig_env_idx - 1;
            end = fig_env_idx;
        }
        BeginTextureMode(fig1);
        ClearBackground(PUFF_BACKGROUND);

        int size = 0;
        for (int i=start; i<end; i++) {
            Table* table = &data.tables[i];
            x = table_col(table, hyper_key[fig_x_idx]);
            y = table_col(table, hyper_key[fig_y_idx]);
            z = use_3d ? table_col(table, hyper_key[fig_z_idx - 1]) : -1;
            if (fig_color_idx != 0) {
                c = table_col(table, hyper_key[fig_color_idx - 1]);
            }
            for (int j=0; j<table->rows; j++) {
                filter[j] = true;
            }
            int filter_param_1 = table_col(table, hyper_key[fig_range1_idx]);
            apply_filter(filter, table, filter_param_1, fig_range1_min_val, fig_range1_max_val);
            int filter_param_2 = table_col(table, hyper_key[fig_range2_idx]);
            apply_filter(filter, table, filter_param_2, fig_range2_min_val, fig_range2_max_val);

            for (int j=0; j<table->rows; j++) {
                if (!filter[j]) {
                    continue;
                }
                points[size] = (Point){
                    table_get(table, j, x),
                    table_get(table, j, y),
                    (z < 0) ? 0.0f : table_get(table, j, z),
                    (fig_color_idx == 0) ? i/(float)data.n : table_get(table, j, c),
                };
                env_indices[size] = (Vector2){i, j};
                size++;
            }
        }
        autoscale(points, size, &args1);
        if (use_3d) {
            CustomUpdateCamera(&args1.camera, fig1_bounds);
        } else {
            args1.mmin[2] = 0.0f;
            args1.mmax[2] = 0.0f;
        }
        toPx(points, glyphs, size, args1);
        Vector2 click = {tooltip.click_x, tooltip.click_y};
        if (right_clicked && CheckCollisionPointRec(click, fig1_bounds)) {
            update_closest(&tooltip, env_indices, glyphs, size, 0, 2*SETTINGS_HEIGHT);
        }
        brighten_hit(&tooltip, env_indices, glyphs, size, 0, 2*SETTINGS_HEIGHT,
            CheckCollisionPointRec(click, fig1_bounds));
        plot_gl(glyphs, size, &shader);
        if (use_3d) {
            BeginMode3D(args1.camera);
            draw_axes3();
            EndMode3D();
        } else {
            draw_axes(args1);
            draw_all_ticks(args1);
        }
        EndTextureMode();

        // Figure 2: hyper boxplot
        args2.scale[0] = fig_box_idx;
        BeginTextureMode(fig2);
        ClearBackground(PUFF_BACKGROUND);
        float bmin[hyper_count];
        float bmax[hyper_count];
        for (int j=0; j<hyper_count; j++) {
            bmin[j] = FLT_MAX;
            bmax[j] = -FLT_MAX;
        }
        for (int i=start; i<end; i++) {
            Table* table = &data.tables[i];
            int f1 = table_col(table, hyper_key[fig_range1_idx]);
            int f2 = table_col(table, hyper_key[fig_range2_idx]);
            for (int k=0; k<table->rows; k++) {
                filter[k] = true;
            }
            apply_filter(filter, table, f1, fig_range1_min_val, fig_range1_max_val);
            apply_filter(filter, table, f2, fig_range2_min_val, fig_range2_max_val);
            for (int j=0; j<hyper_count; j++) {
                int col = table_col(table, hyper_key[j]);
                if (col < 0) {
                    continue;
                }
                for (int k=0; k<table->rows; k++) {
                    if (!filter[k]) {
                        continue;
                    }
                    float val = table_get(table, k, col);
                    bmin[j] = fminf(bmin[j], val);
                    bmax[j] = fmaxf(bmax[j], val);
                }
            }
        }
        float gmin = FLT_MAX, gmax = -FLT_MAX;
        for (int j=0; j<hyper_count; j++) {
            if (bmin[j] > bmax[j] || (args2.scale[0] == LOG && bmax[j] <= 0)) {
                continue;
            }
            float lo = bmin[j] > 0 ? bmin[j] : bmax[j];
            gmin = fminf(gmin, lo);
            gmax = fmaxf(gmax, bmax[j]);
        }
        if (gmin > gmax) {
            gmin = 1e-5f;
            gmax = 1e5f;
        } else if (gmin == gmax) {
            gmin = args2.scale[0] == LOG ? gmin/10 : gmin - 1;
            gmax = args2.scale[0] == LOG ? gmax*10 : gmax + 1;
        }
        args2.mmin[0] = args2.scale[0] == LOGIT ? 0.5f : gmin;
        args2.mmax[0] = args2.scale[0] == LOGIT ? 0.999f : gmax;
        int bsize = 0;
        float plot_width = args2.width - args2.left_margin - args2.right_margin;
        float plot_height = args2.height - args2.top_margin - args2.bottom_margin;
        float dy = plot_height / (float)hyper_count;
        float x_min = scale_val(args2.scale[0], args2.mmin[0]);
        float x_max = scale_val(args2.scale[0], args2.mmax[0]);
        Color bar = Fade(PUFF_WHITE, 0.08f);
        for (int j=0; j<hyper_count; j++) {
            if (bmin[j] > bmax[j]) {
                continue;
            }
            float lo = scale_val(args2.scale[0], bmin[j]);
            float hi = scale_val(args2.scale[0], bmax[j]);
            float left = args2.left_margin + (lo - x_min)/(x_max - x_min)*plot_width;
            float right = args2.left_margin + (hi - x_min)/(x_max - x_min)*plot_width;
            left = fminf(fmaxf(left, args2.left_margin), args2.width - args2.right_margin);
            right = fmaxf(fminf(right, args2.width - args2.right_margin), args2.left_margin);
            if (right - left < 2) {
                right = left + 2;
            }
            DrawRectangle(left, args2.top_margin + j*dy, right - left, dy, bar);
        }
        for (int i=start; i<end; i++) {
            Table* table = &data.tables[i];
            int f1 = table_col(table, hyper_key[fig_range1_idx]);
            int f2 = table_col(table, hyper_key[fig_range2_idx]);
            for (int k=0; k<table->rows; k++) {
                filter[k] = true;
            }
            apply_filter(filter, table, f1, fig_range1_min_val, fig_range1_max_val);
            apply_filter(filter, table, f2, fig_range2_min_val, fig_range2_max_val);
            Color dc = rgb(i / (float)data.n);
            for (int j=0; j<hyper_count; j++) {
                int col = table_col(table, hyper_key[j]);
                if (col < 0) {
                    continue;
                }
                float y0 = args2.top_margin + j*dy;
                for (int k=0; k<table->rows; k++) {
                    if (!filter[k]) {
                        continue;
                    }
                    float sval = scale_val(args2.scale[0], table_get(table, k, col));
                    float hx = args2.left_margin + (sval - x_min)/(x_max - x_min)*plot_width;
                    hx = fminf(fmaxf(hx, args2.left_margin), args2.width - args2.right_margin);
                    float hy = y0 + 2 + fmodf((k + 1)*0.618034f, 1.0f)*fmaxf(dy - 4, 1);
                    glyphs[bsize] = (Glyph){
                        hx, hy, (float)bsize,
                        dc.r/255.0f, dc.g/255.0f, dc.b/255.0f, dc.a/255.0f
                    };
                    env_indices[bsize] = (Vector2){i, k};
                    bsize++;
                }
            }
        }
        if (right_clicked && CheckCollisionPointRec(click, fig2_bounds)) {
            update_closest(&tooltip, env_indices, glyphs, bsize,
                fig1.texture.width, 2*SETTINGS_HEIGHT);
        }
        brighten_hit(&tooltip, env_indices, glyphs, bsize,
            fig1.texture.width, 2*SETTINGS_HEIGHT, false);
        if (bsize > 0) {
            plot_gl(glyphs, bsize, &shader);
        }
        draw_axes(args2);
        draw_box_ticks(hyper_label, hyper_count, args2);
        EndTextureMode();

        DrawTextureRec(
            fig1.texture,
            (Rectangle){0, 0, fig1.texture.width, -fig1.texture.height },
            (Vector2){ 0, 2*SETTINGS_HEIGHT }, WHITE
        );
        BeginShaderMode(blur_shader);
        rlSetBlendMode(RL_BLEND_ADDITIVE);
        DrawTextureRec(
            fig1_overlay.texture,
            (Rectangle){0, 0, fig1_overlay.texture.width, -fig1_overlay.texture.height },
            (Vector2){ 0, 2*SETTINGS_HEIGHT }, WHITE
        );
        rlSetBlendMode(RL_BLEND_ALPHA);
        EndShaderMode();
        DrawTextureRec(
            fig2.texture,
            (Rectangle){ 0, 0, fig2.texture.width, -fig2.texture.height },
            (Vector2){ fig1.texture.width, 2*SETTINGS_HEIGHT }, WHITE
        );

        // UI
        float y = SEP + SETTINGS_HEIGHT/2.0f - MeasureTextEx(args1.font_small, "Env", args1.axis_tick_font_size, 0).y/2.0f;
        float x = SEP;
        DrawTextEx(args1.font_small, "Env", (Vector2){x, y}, args1.axis_tick_font_size, 0, WHITE);
        x += MeasureTextEx(args1.font_small, "Env", args1.axis_tick_font_size, 0).x + SEP;

        Rectangle fig_env_rect = {x, SEP, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        x += DROPDOWN_WIDTH + SPACER;
        if (GuiDropdownBox(fig_env_rect, env_options, &fig_env_idx, fig_env_active)){
            fig_env_active = !fig_env_active;
        }

        // X axis
        DrawTextEx(args1.font_small, "X", (Vector2){x, y}, args1.axis_tick_font_size, 0, RED);
        x += MeasureTextEx(args1.font_small, "X", args1.axis_tick_font_size, 0).x + SEP;

        Rectangle fig_x_rect = {x, SEP, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        x += DROPDOWN_WIDTH;
        if (GuiDropdownBox(fig_x_rect, options, &fig_x_idx, fig_x_active)){
            fig_x_active = !fig_x_active;
        }
        Rectangle fig_xscale_rect = {x, SEP, TOGGLE_WIDTH, SETTINGS_HEIGHT};
        x += TOGGLE_WIDTH + SPACER;
        if (GuiDropdownBox(fig_xscale_rect, scale_options, &args1.scale[0], fig_xscale_active)){
            fig_xscale_active = !fig_xscale_active;
        }

        // Y axis
        DrawTextEx(args1.font_small, "Y", (Vector2){x, y}, args1.axis_tick_font_size, 0, GREEN);
        x += MeasureTextEx(args1.font_small, "Y", args1.axis_tick_font_size, 0).x + SEP;

        Rectangle fig_y_rect = {x, SEP, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        x += DROPDOWN_WIDTH;
        if (GuiDropdownBox(fig_y_rect, options, &fig_y_idx, fig_y_active)){
            fig_y_active = !fig_y_active;
        }
        Rectangle fig_yscale_rect = {x, SEP, TOGGLE_WIDTH, SETTINGS_HEIGHT};
        x += TOGGLE_WIDTH + SPACER;
        if (GuiDropdownBox(fig_yscale_rect, scale_options, &args1.scale[1], fig_yscale_active)){
            fig_yscale_active = !fig_yscale_active;
        }

        // Z axis
        DrawTextEx(args1.font_small, "Z", (Vector2){x, y}, args1.axis_tick_font_size, 0, BLUE);
        x += MeasureTextEx(args1.font_small, "Z", args1.axis_tick_font_size, 0).x + SEP;

        Rectangle fig_z_rect = {x, SEP, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        x += DROPDOWN_WIDTH;
        if (GuiDropdownBox(fig_z_rect, z_options, &fig_z_idx, fig_z_active)){
            fig_z_active = !fig_z_active;
        }
        Rectangle fig_zscale_rect = {x, SEP, TOGGLE_WIDTH, SETTINGS_HEIGHT};
        x += TOGGLE_WIDTH + SPACER;
        if (GuiDropdownBox(fig_zscale_rect, scale_options, &args1.scale[2], fig_zscale_active)){
            fig_zscale_active = !fig_zscale_active;
        }

        // Color
        DrawTextEx(args1.font_small, "C", (Vector2){x, y}, args1.axis_tick_font_size, 0, WHITE);
        x += MeasureTextEx(args1.font_small, "C", args1.axis_tick_font_size, 0).x + SEP;

        Rectangle fig_color_rect = {x, SEP, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        x += DROPDOWN_WIDTH;
        if (GuiDropdownBox(fig_color_rect, env_hyper_options, &fig_color_idx, fig_color_active)){
            fig_color_active = !fig_color_active;
        }
        Rectangle fig_colorscale_rect = {x, SEP, TOGGLE_WIDTH, SETTINGS_HEIGHT};
        x += TOGGLE_WIDTH + SPACER;
        if (GuiDropdownBox(fig_colorscale_rect, scale_options, &args1.scale[3], fig_colorscale_active)){
            fig_colorscale_active = !fig_colorscale_active;
        }

        // Filters
        DrawTextEx(args1.font_small, "F1", (Vector2){x, y}, args1.axis_tick_font_size, 0, WHITE);
        x += MeasureTextEx(args1.font_small, "F1", args1.axis_tick_font_size, 0).x + SEP;

        GuiDropdownFilter(x, SEP, options,
                &fig_range1_idx, &fig_range1_active, focus, fig_range1_min,
                &fig_range1_min_val, fig_range1_max, &fig_range1_max_val);
        x += DROPDOWN_WIDTH + 2*TOGGLE_WIDTH + SPACER;

        DrawTextEx(args1.font_small, "F2", (Vector2){x, y}, args1.axis_tick_font_size, 0, WHITE);
        x += MeasureTextEx(args1.font_small, "F2", args1.axis_tick_font_size, 0).x + SEP;

        GuiDropdownFilter(x, SEP, options,
            &fig_range2_idx, &fig_range2_active, focus, fig_range2_min,
            &fig_range2_min_val, fig_range2_max, &fig_range2_max_val);
        x += DROPDOWN_WIDTH + 2*TOGGLE_WIDTH + SPACER;

        // Box
        DrawTextEx(args1.font_small, "Box", (Vector2){x, y}, args1.axis_tick_font_size, 0, WHITE);
        x += MeasureTextEx(args1.font_small, "Box", args1.axis_tick_font_size, 0).x + SEP;

        Rectangle box_rect = {x, SEP, TOGGLE_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(box_rect, scale_options, &fig_box_idx, fig_box_active)) {
            fig_box_active = !fig_box_active;
        }

        // Puffer
        float width = GetScreenWidth();
        DrawTexturePro(
            puffer,
            (Rectangle){0, 128, 128, 128},
            (Rectangle){width - 48, -8, 48, 48},
            (Vector2){0, 0},
            0,
            WHITE
        );

        // Tooltip
        int env_idx = tooltip.env_idx;
        int ary_idx = tooltip.ary_idx;
        Table* table = &data.tables[env_idx];
        char* env_key = table->name;

        float cost = table_get(table, ary_idx, table_col(table, "uptime"));
        float score = table_get(table, ary_idx, table_col(table, "env/score"));
        float steps = table_get(table, ary_idx, table_col(table, "agent_steps"));
        if (tooltip.active) {
            const char* text = TextFormat("%s\nscore = %f\ncost = %f\nsteps = %f", env_key, score, cost, steps);
            Vector2 text_size = MeasureTextEx(args1.font_small, text, args1.axis_tick_font_size, 0);
            float x = tooltip.x;
            float y = tooltip.y;
            if (x + text_size.x + 4 > GetScreenWidth()) {
                x = x - text_size.x - 4;
            }
            if (y + text_size.y + 4 > GetScreenHeight()) {
                y = y - text_size.y - 4;
            }
            DrawRectangle(x, y, text_size.x + 4, text_size.y + 4, PUFF_BACKGROUND);
            DrawTextEx(args1.font_small, text, (Vector2){x + 2, y + 2}, args1.axis_tick_font_size, 0, WHITE);
        }
        EndDrawing();

        // Copy hypers to clipboard
        if (right_clicked) {
            copy_hypers_to_clipboard(table, clipboard, ary_idx);
        }
    }

    // Cleanup
    for (int i = 0; i < data.n; i++) {
        table_free(&data.tables[i]);
    }
    free(data.tables);
    free(options);
    free(env_hyper_options);
    free(z_options);
    free(env_options);
    free(clipboard);
    free(points);
    free(glyphs);
    free(env_indices);
    free(filter);

    // Raylib resources
    UnloadShader(shader);
    UnloadShader(blur_shader);
    UnloadRenderTexture(fig1);
    UnloadRenderTexture(fig1_overlay);
    UnloadRenderTexture(fig2);
    CloseWindow();
    return 0;
}

#endif
