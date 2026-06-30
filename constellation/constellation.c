#include <float.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "cJSON.h"
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
void CustomUpdateCamera(Camera *camera, float orbitSpeed) {
    float cameraOrbitalSpeed = CAMERA_ORBITAL_SPEED*GetFrameTime();
    Matrix rotation = MatrixRotate(GetCameraUp(camera), cameraOrbitalSpeed);
    Vector3 view = Vector3Subtract(camera->position, camera->target);
    view = Vector3Transform(view, rotation);
    camera->position = Vector3Add(camera->target, view);
    CameraMoveToTarget(camera, -GetMouseWheelMove());
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

int hyper_count = 25;
char *hyper_key[25] = {
    "agent_steps",
    "uptime",
    "env/perf",
    "env/score",
    "train/learning_rate",
    "train/ent_coef",
    "train/gamma",
    "train/gae_lambda",
    "train/vtrace_rho_clip",
    "train/vtrace_c_clip",
    "train/clip_coef",
    "train/vf_clip_coef",
    "train/vf_coef",
    "train/max_grad_norm",
    "train/beta1",
    "train/beta2",
    "train/eps",
    "train/prio_alpha",
    "train/prio_beta0",
    "train/horizon",
    "train/replay_ratio",
    "train/minibatch_size",
    "policy/hidden_size",
    "policy/num_layers",
    "vec/total_agents",
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
    char *key;
    float *ary;
    int n;
} Hyper;

typedef struct {
    char *key;
    Hyper *hypers;
    int n;
} Env;

typedef struct {
    Env *envs;
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


Hyper* get_hyper(Dataset *data, char *env, char* hyper) {
    for (int i = 0; i < data->n; i++) {
        if (strcmp(data->envs[i].key, env) != 0) {
            continue;
        }
        for (int j = 0; j < data->envs[i].n; j++) {
            if (strcmp(data->envs[i].hypers[j].key, hyper) == 0) {
                return &data->envs[i].hypers[j];
            }
        }
    }
    printf("Error: hyper %s not found in env %s\n", hyper, env);
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

void draw_box_ticks(char* hypers[], int hyper_count, PlotArgs args) {
    Vector2 tick_n = compute_ticks(args);
    char x_ticks[(int)tick_n.x][32];
    label_ticks(x_ticks, args, 0, tick_n.x);
    char fixed_hypers[hyper_count][32];
    for (int i=0; i<hyper_count; i++) {
        strncpy(fixed_hypers[hyper_count - i - 1], hypers[i], 32);
    }
    draw_ticks(x_ticks, tick_n.x, fixed_hypers, hyper_count, args);
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

void boxplot(Hyper* hyper, int x_scale, int i, int hyper_count, PlotArgs args, Color color, bool* filter) {
    int width = args.width;
    int height = args.height;

    float plot_width = width - args.left_margin - args.right_margin;
    float plot_height = height - args.top_margin - args.bottom_margin;

    float x_min = scale_val(x_scale, args.mmin[0]);
    float x_max = scale_val(x_scale, args.mmax[0]);

    float dy = plot_height/((float)hyper_count);

    float* ary = hyper->ary;
    float mmin = ary[0];
    float mmax = ary[0];
    for (int j=0; j<hyper->n; j++) {
        if (filter != NULL && !filter[j]) {
            continue;
        }
        mmin = fmin(mmin, ary[j]);
        mmax = fmax(mmax, ary[j]);
    }

    mmin = scale_val(x_scale, mmin);
    mmax = scale_val(x_scale, mmax);

    float left = args.left_margin + (mmin - x_min)/(x_max - x_min)*plot_width;
    float right = args.left_margin + (mmax - x_min)/(x_max - x_min)*plot_width;

    // TODO - rough patch
    left = fminf(fmax(left, args.left_margin), width - args.right_margin);
    right = fmaxf(fmin(right, width - args.right_margin), 0);
    DrawRectangle(left, args.top_margin + i*dy, right - left, dy, color);
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

void apply_filter(bool* filter, Hyper* param, float min, float max) {
    for (int i=0; i<param->n; i++) {
        float val = param->ary[i];
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

void copy_hypers_to_clipboard(Env *env, char* buffer, int ary_idx) {
    char* start = buffer;
    char* prefix = NULL;
    int prefix_len = 0;
    for (int hyper_idx = 0; hyper_idx < env->n; hyper_idx++) {
        Hyper *hyper = &env->hypers[hyper_idx];
        char *slash = strchr(hyper->key, '/');
        if (!slash || ary_idx >= hyper->n) {
            continue;
        }

        if (prefix == NULL || strncmp(prefix, hyper->key, prefix_len) != 0) {
            if (prefix != NULL) {
                buffer += sprintf(buffer, "\n");
            }
            prefix = hyper->key;
            prefix_len = slash - prefix;
            buffer += sprintf(buffer, "[");
            snprintf(buffer, prefix_len+1, "%s", prefix);
            buffer += prefix_len;
            buffer += sprintf(buffer, "]\n");
        }

        char* suffix = slash + 1;
        double val = hyper->ary[ary_idx];
        if (strcmp(suffix, "total_timesteps") == 0) {
            // Use agent_steps (training-only) instead of total_timesteps (train+eval)
            for (int k = 0; k < env->n; k++) {
                if (strcmp(env->hypers[k].key, "agent_steps") == 0) {
                    val = env->hypers[k].ary[ary_idx];
                    break;
                }
            }
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

//strof bottlenecks loads
float fast_atof(char **s) {
    char *p = *s;
    float sign = 1.0f;
    if (*p == '-') {
        sign = -1.0f; p++;
    }
    float val = 0.0f;
    while (*p >= '0' && *p <= '9') {
        val = val * 10.0f + (*p++ - '0');
    }
    if (*p == '.') {
        p++;
        float frac = 0.1f;
        while (*p >= '0' && *p <= '9') {
            val += (*p++ - '0') * frac; frac *= 0.1f;
        }
    }
    if (*p == 'e' || *p == 'E') {
      p++;
      int esign = 1;
      if (*p == '-') {
          esign = -1; p++;
      } else if (*p == '+') {
          p++;
      }
      int exp = 0;
      while (*p >= '0' && *p <= '9') {
          exp = exp * 10 + (*p++ - '0');
      }
      val *= powf(10.0f, esign * exp);
  }
  *s = p;
  return sign * val;
}

int main(void) {
    FILE *file = fopen("resources/constellation/experiments.json", "r");
    if (!file) {
        printf("Error opening file\n");
        return 1;
    }

    // Read in file
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *json_str = malloc(file_size + 1);
    fread(json_str, 1, file_size, file);
    json_str[file_size] = '\0';
    fclose(file);
    cJSON *root = cJSON_Parse(json_str);
    if (!root) {
        printf("JSON parse error: %.100s\n", cJSON_GetErrorPtr());
        free(json_str);
        return 1;
    }
    if (!cJSON_IsObject(root)) {
        printf("Error: Root is not an object\n");
        return 1;
    }

    // Load in dataset
    Dataset data = {NULL, 0};
    cJSON *json_env = root->child;
    while (json_env) {
        data.n++;
        json_env = json_env->next;
    }

    Env *envs = calloc(data.n, sizeof(Env));
    data.envs = envs;
    json_env = root->child;
    int max_data_points = 0;
    for (int i=0; i<data.n; i++) {
        cJSON *json_hyper = json_env->child;
        while (json_hyper) {
            envs[i].n++;
            json_hyper = json_hyper->next;
        }
        envs[i].key = json_env->string;
        envs[i].hypers = calloc(envs[i].n, sizeof(Hyper));
        json_hyper = json_env->child;
        for (int j=0; j<envs[i].n; j++, json_hyper=json_hyper->next) {
            envs[i].hypers[j].key = json_hyper->string;
            int capacity = 1;
            for (char* p = json_hyper->valuestring; *p; p++) {
                if (*p == ',') {
                    capacity++;
                }
            }
            if (capacity > max_data_points) {
                max_data_points = capacity;
            }
            envs[i].hypers[j].ary = calloc(capacity, sizeof(float));

            int n = 0;
            char* s = json_hyper->valuestring;
            while (*s) {
                envs[i].hypers[j].ary[n++] = fast_atof(&s);
                if (*s == ',') {
                    s++;
                }
            }
            envs[i].hypers[j].n = n;
        }
        json_env = json_env->next;
    }
    int total_points = 0;
    for (int i=0; i<data.n; i++) {
        total_points += envs[i].hypers[0].n;
    }

    // Create options as a semicolon-separated string
    size_t options_len = 0;
    for (int i = 0; i < hyper_count; i++) {
        options_len += strlen(hyper_key[i]) + 1;
    }
    char *options = malloc(options_len);
    options[0] = '\0';
    for (int i = 0; i < hyper_count; i++) {
        if (i > 0) strcat(options, ";");
        strcat(options, hyper_key[i]);
    }

    // Options with extra "env_name;"
    char* extra = "env_name;";
    char *env_hyper_options = malloc(options_len + strlen(extra));
    strcpy(env_hyper_options, extra);
    strcat(env_hyper_options, options);

    // Env names as semi-colon-separated string
    size_t env_options_len = 4;
    for (int i = 0; i < data.n; i++) {
        env_options_len += strlen(data.envs[i].key) + 1;
    }
    char *env_options = malloc(env_options_len);
    strcpy(env_options, "all;");
    env_options[4] = '\0';
    for (int i = 0; i < data.n; i++) {
        if (i > 0) strcat(env_options, ";");
        strcat(env_options, data.envs[i].key);
    }

    char* clipboard = malloc(16384);

    // Points
    printf("total points: %d", total_points);
    Point* points = calloc(total_points, sizeof(Point));
    Glyph* glyphs = calloc(total_points, sizeof(Glyph));
    Vector2* env_indices = calloc(total_points, sizeof(Vector2));

    // Initialize Raylib
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(2*DEFAULT_PLOT_ARGS.width, 2*DEFAULT_PLOT_ARGS.height + 2*SETTINGS_HEIGHT, "Puffer Constellation");
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
    RenderTexture2D fig1 = LoadRenderTexture(args1.width, args1.height);
    RenderTexture2D fig1_overlay = LoadRenderTexture(args1.width, args1.height);
    int fig_env_idx = 0;
    bool fig_env_active = false;
    bool fig_x_active = false;
    int fig_x_idx = 1;
    bool fig_xscale_active = false;
    bool fig_y_active = false;
    int fig_y_idx = 2;
    bool fig_yscale_active = false;
    bool fig_z_active = false;
    int fig_z_idx = 0;
    bool fig_zscale_active = false;
    int fig_color_idx = 0;
    bool fig_color_active = false;
    bool fig_colorscale_active = false;
    bool fig_range1_active = false;
    int fig_range1_idx = 2;
    char fig_range1_min[32] = {0};
    char fig_range1_max[32] = {0};
    float fig_range1_min_val = 0;
    float fig_range1_max_val = FLT_MAX;
    bool fig_range2_active = false;
    int fig_range2_idx = 1;
    char fig_range2_min[32] = {0};
    char fig_range2_max[32] = {0};
    float fig_range2_min_val = FLT_MIN;
    float fig_range2_max_val = FLT_MAX;
    int fig_box_idx = LOG;
    bool fig_box_active = false;

    char* scale_options = "linear;log;logit";

    PlotArgs args2 = DEFAULT_PLOT_ARGS;
    RenderTexture2D fig2 = LoadRenderTexture(args2.width, args2.height);
    args2.right_margin = 50;
    args2.scale[0] = 1;

    PlotArgs args3 = DEFAULT_PLOT_ARGS;
    RenderTexture2D fig3 = LoadRenderTexture(args3.width, args3.height);
    RenderTexture2D fig3_overlay = LoadRenderTexture(args1.width, args1.height);
    args3.left_margin = 10;
    args3.right_margin = 10;
    args3.top_margin = 10;
    args3.bottom_margin = 10;
    args3.x_label = "tsne1";
    args3.y_label = "tsne2";

    PlotArgs args4 = DEFAULT_PLOT_ARGS;
    RenderTexture2D fig4 = LoadRenderTexture(args4.width, args4.height);
    args4.x_label = "Value";
    args4.y_label = "Hyperparameter";
    args4.left_margin = 170;
    args4.right_margin = 50;
    args4.top_margin = 10;
    args4.bottom_margin = 50;

    Hyper* x;
    Hyper* y;
    Hyper* z;
    Hyper* c;
    char* x_label;
    char* y_label;
    char* z_label;

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

        // Figure 1
        x_label = hyper_key[fig_x_idx];
        y_label = hyper_key[fig_y_idx];
        z_label = hyper_key[fig_z_idx];
        args1.x_label = x_label;
        args1.y_label = y_label;
        args1.z_label = z_label;
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
            char* env = data.envs[i].key;
            x = get_hyper(&data, env, hyper_key[fig_x_idx]);
            y = get_hyper(&data, env, hyper_key[fig_y_idx]);
            z = get_hyper(&data, env, hyper_key[fig_z_idx]);
            if (fig_color_idx != 0) {
                c = get_hyper(&data, env, hyper_key[fig_color_idx - 1]);
            }
            for (int j=0; j<x->n; j++) {
                filter[j] = true;
            }
            Hyper* filter_param_1 = get_hyper(&data, env, hyper_key[fig_range1_idx]);
            apply_filter(filter, filter_param_1, fig_range1_min_val, fig_range1_max_val);
            Hyper* filter_param_2 = get_hyper(&data, env, hyper_key[fig_range2_idx]);
            apply_filter(filter, filter_param_2, fig_range2_min_val, fig_range2_max_val);

            for (int j=0; j<x->n; j++) {
                if (!filter[j]) {
                    continue;
                }
                points[size] = (Point){
                    x->ary[j],
                    y->ary[j],
                    z->ary[j],
                    (fig_color_idx == 0) ? i/(float)data.n : c->ary[j],
                };
                env_indices[size] = (Vector2){i, j};
                size++;
            }
        }
        autoscale(points, size, &args1);
        toPx(points, glyphs, size, args1);
        update_closest(&tooltip, env_indices, glyphs, size, 0, 2*SETTINGS_HEIGHT);
        plot_gl(glyphs, size, &shader);

        BeginMode3D(args1.camera);
        CustomUpdateCamera(&args1.camera, CAMERA_ORBITAL_SPEED);
        draw_axes3();
        EndMode3D();
        EndTextureMode();

        // Figure 2
        x_label = hyper_key[fig_x_idx];
        y_label = hyper_key[fig_y_idx];
        args2.scale[0] = args1.scale[0];
        args2.scale[1] = args1.scale[1];
        args2.x_label = x_label;
        args2.y_label = y_label;
        args2.top_margin = 20;
        args2.left_margin = 100;
        BeginTextureMode(fig2);
        ClearBackground(PUFF_BACKGROUND);

        autoscale(points, size, &args2);
        args2.mmin[2] = 0.0f;
        args2.mmax[2] = 0.0f;
        toPx(points, glyphs, size, args2);
        update_closest(&tooltip, env_indices, glyphs, size, fig1.texture.width, 2*SETTINGS_HEIGHT);
        plot_gl(glyphs, size, &shader);
        draw_axes(args2);
        draw_all_ticks(args2);
        EndTextureMode();

        // Figure 3
        BeginTextureMode(fig3);
        ClearBackground(PUFF_BACKGROUND);
        size = 0;
        for (int i=0; i<data.n; i++) {
            char* env = data.envs[i].key;
            x = get_hyper(&data, env, "tsne1");
            y = get_hyper(&data, env, "tsne2");
            for (int j=0; j<x->n; j++) {
                filter[j] = true;
            }
            Hyper* filter_param_1 = get_hyper(&data, env, hyper_key[fig_range1_idx]);
            apply_filter(filter, filter_param_1, fig_range1_min_val, fig_range1_max_val);
            Hyper* filter_param_2 = get_hyper(&data, env, hyper_key[fig_range2_idx]);
            apply_filter(filter, filter_param_2, fig_range2_min_val, fig_range2_max_val);

            for (int j=0; j<x->n; j++) {
                if (!filter[j]) {
                    continue;
                }
                points[size] = (Point){
                    x->ary[j],
                    y->ary[j],
                    0.0f,
                    i/(float)data.n
                };
                env_indices[size] = (Vector2){i, j};
                size++;
            }
        }
        autoscale(points, size, &args3);
        toPx(points, glyphs, size, args3);
        update_closest(&tooltip, env_indices, glyphs, size, 0, fig1.texture.height + 2*SETTINGS_HEIGHT);
        plot_gl(glyphs, size, &shader);

        //draw_axes(args3);
        EndTextureMode();

        // Figure 4
        args4.scale[0] = fig_box_idx;
        if (args4.scale[0] == LINEAR) {
            args4.mmin[0] = 0.0f;
            args4.mmax[0] = 5.0f;
        } else if (args4.scale[0] == LOG) {
            args4.mmin[0] = 1.0e-5f;
            args4.mmax[0] = 1.0e5f;
        } else if (args4.scale[0] == LOGIT) {
            args4.mmin[0] = 0.5f;
            args4.mmax[0] = 0.999f;
        }
        BeginTextureMode(fig4);
        ClearBackground(PUFF_BACKGROUND);
        rlSetBlendFactorsSeparate(0x0302, 0x0303, 1, 0x0303, 0x8006, 0x8006);
        BeginBlendMode(BLEND_CUSTOM_SEPARATE);
        Color color = Fade(PUFF_CYAN, 1.0f / (float)(end - start));
        for (int i=start; i<end; i++) {
            Env* env = &data.envs[i];
            Hyper* filter_param_1 = get_hyper(&data, env->key, hyper_key[fig_range1_idx]);
            Hyper* filter_param_2 = get_hyper(&data, env->key, hyper_key[fig_range2_idx]);
            for (int j=0; j<hyper_count; j++) {
                Hyper* hyper = get_hyper(&data, env->key, hyper_key[j]);
                for (int k=0; k<hyper->n; k++) {
                    filter[k] = true;
                }
                apply_filter(filter, filter_param_1, fig_range1_min_val, fig_range1_max_val);
                apply_filter(filter, filter_param_2, fig_range2_min_val, fig_range2_max_val);
                boxplot(hyper, args4.scale[0], j, hyper_count, args4, color, filter);
            }
        }
        EndBlendMode();
        draw_axes(args4);
        draw_box_ticks(hyper_key, hyper_count, args4);
        EndTextureMode();

        // Figure 1-4
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
        DrawTextureRec(
            fig3.texture,
            (Rectangle){ 0, 0, fig3.texture.width, -fig3.texture.height },
            (Vector2){ 0, 2*SETTINGS_HEIGHT + fig1.texture.height }, WHITE
        );
        BeginShaderMode(blur_shader);
        rlSetBlendMode(RL_BLEND_ADDITIVE);
        DrawTextureRec(
            fig3_overlay.texture,
            (Rectangle){0, 0, fig3_overlay.texture.width, -fig3_overlay.texture.height },
            (Vector2){ 0, 2*SETTINGS_HEIGHT + fig1.texture.height }, WHITE
        );
        rlSetBlendMode(RL_BLEND_ALPHA);
        EndShaderMode();
        DrawTextureRec(
            fig4.texture,
            (Rectangle){ 0, 0, fig4.texture.width, -fig4.texture.height },
            (Vector2){ fig1.texture.width, fig1.texture.height + 2*SETTINGS_HEIGHT }, WHITE
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
        if (GuiDropdownBox(fig_z_rect, options, &fig_z_idx, fig_z_active)){
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

        // Temp hack
        args2.scale[3] = args1.scale[3];
        args3.scale[3] = args1.scale[3];
        args4.scale[3] = args1.scale[3];

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
        Env* env = &data.envs[env_idx];
        char* env_key = env->key;

        float cost = get_hyper(&data, env_key, "uptime")->ary[ary_idx];
        float score = get_hyper(&data, env_key, "env/score")->ary[ary_idx];
        float steps = get_hyper(&data, env_key, "agent_steps")->ary[ary_idx];
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
            DrawCircle(tooltip.x, tooltip.y, 2, PUFF_CYAN);
            DrawTextEx(args1.font_small, text, (Vector2){x + 2, y + 2}, args1.axis_tick_font_size, 0, WHITE);
        }
        EndDrawing();

        // Copy hypers to clipboard
        if (right_clicked) {
            copy_hypers_to_clipboard(env, clipboard, ary_idx);
        }
    }

    // Cleanup
    for (int i = 0; i < data.n; i++) {
        for (int j = 0; j < envs[i].n; j++) {
            free(envs[i].hypers[j].ary);
        }
        free(envs[i].hypers);
    }
    free(envs);
    cJSON_Delete(root);
    free(json_str);
    free(options);
    free(env_hyper_options);
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
    UnloadRenderTexture(fig3);
    UnloadRenderTexture(fig3_overlay);
    UnloadRenderTexture(fig4);
    CloseWindow();
    return 0;
}
